#include <glib.h>
#include <stdio.h>
#include <string.h>
#include <ocl.h>

typedef struct {
    gboolean could_allocate;
    gboolean could_write;
    gboolean could_read;
    double alloc_time;
    double warmup_time;
    double write_time;
    double read_time;
} Result;


static void
measure_allocation (cl_context context, cl_command_queue queue, cl_mem_flags flags, size_t size, Result *result)
{
    cl_mem mem;
    cl_int err;
    GTimer *timer;

    timer = g_timer_new ();
    memset (result, 0, sizeof (Result));

    g_timer_start (timer);
    mem = clCreateBuffer (context, flags, size, NULL, &err);
    g_timer_stop (timer);
    result->alloc_time = g_timer_elapsed (timer, NULL);
    result->could_allocate = err == CL_SUCCESS;

    if (result->could_allocate) {
        char *data;

        data = malloc (size);

        g_timer_start (timer);
        err = clEnqueueWriteBuffer (queue, mem, CL_TRUE, 0, size, data, 0, NULL, NULL);
        g_timer_stop (timer);
        result->warmup_time = g_timer_elapsed (timer, NULL);

        g_timer_start (timer);
        err = clEnqueueWriteBuffer (queue, mem, CL_TRUE, 0, size, data, 0, NULL, NULL);
        g_timer_stop (timer);
        result->write_time = g_timer_elapsed (timer, NULL);
        result->could_write = err == CL_SUCCESS;

        g_timer_start (timer);
        err = clEnqueueReadBuffer (queue, mem, CL_TRUE, 0, size, data, 0, NULL, NULL);
        g_timer_stop (timer);
        result->read_time = g_timer_elapsed (timer, NULL);
        result->could_read = err == CL_SUCCESS;

        free (data);
    }

    OCL_CHECK_ERROR (clReleaseMemObject (mem));
    g_timer_destroy (timer);
}

static void
print_timing (Result *result, size_t size)
{
    static char *bools[] = {"no", "yes"};

    g_print ("    Could allocate             : %s (%3.5f s)\n"
             "    Could write                : %s (%3.5f s, %3.2f MB/s, warm up: %3.5fs)\n"
             "    Could read                 : %s (%3.5f s, %3.2f MB/s)\n",
             bools[result->could_allocate], result->alloc_time,
             bools[result->could_write], result->write_time, size / result->write_time / 1024. / 1024., result->warmup_time,
             bools[result->could_read], result->read_time, size / result->read_time / 1024. / 1024.);
}

int
main (int argc, const char **argv)
{
    OclPlatform *ocl;
    cl_device_id *devices;
    int num_devices;

    ocl = ocl_new_from_args_bare (argc, argv);

    num_devices = ocl_get_num_devices (ocl);
    devices = ocl_get_devices (ocl);

    for (int i = 0; i < num_devices; i++) {
        char name[256];
        cl_context context;
        cl_command_queue queue;
        cl_ulong global_mem_size;
        cl_ulong max_mem_alloc_size;
        cl_int err;
        Result result;

        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_NAME, 256, name, NULL));
        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof (cl_ulong), &global_mem_size, NULL));
        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof (cl_ulong), &max_mem_alloc_size, NULL));

        g_print ("%s\n"
                 "  CL_DEVICE_GLOBAL_MEM_SIZE    : %-11lu B (%3.2f MB)\n"
                 "  CL_DEVICE_MAX_MEM_ALLOC_SIZE : %-11lu B (%3.2f MB, %3.1f%%)\n",
                 name,
                 global_mem_size, global_mem_size / 1024. / 1024.,
                 max_mem_alloc_size, max_mem_alloc_size / 1024. / 1024.,
                 ((double) max_mem_alloc_size) / global_mem_size * 100);

        context = clCreateContext (NULL, 1, &devices[i], NULL, NULL, &err);
        queue = clCreateCommandQueue (context, devices[i], CL_QUEUE_PROFILING_ENABLE, &err);

        OCL_CHECK_ERROR (err);

        measure_allocation (context, queue, CL_MEM_READ_WRITE, max_mem_alloc_size, &result);
        g_print ("  CL_MEM_READ_WRITE\n");
        print_timing (&result, max_mem_alloc_size);

        measure_allocation (context, queue, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, max_mem_alloc_size, &result);
        g_print ("  CL_MEM_ALLOC_HOST_PTR\n");
        print_timing (&result, max_mem_alloc_size);

        OCL_CHECK_ERROR (clReleaseCommandQueue (queue));
        OCL_CHECK_ERROR (clReleaseContext (context));

        if (i < num_devices - 1)
            g_print ("\n");
    }

    ocl_free (ocl);
}
