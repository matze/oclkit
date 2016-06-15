#include <glib.h>
#include <stdio.h>
#include <ocl.h>


int
main (int argc, const char **argv)
{
    OclPlatform *ocl;
    cl_device_id *devices;
    GTimer *timer;
    int num_devices;
    static char *bools[] = {"no", "yes"};

    ocl = ocl_new_from_args_bare (argc, argv);

    num_devices = ocl_get_num_devices (ocl);
    devices = ocl_get_devices (ocl);
    timer = g_timer_new ();

    for (int i = 0; i < num_devices; i++) {
        char name[256];
        cl_context context;
        cl_command_queue queue;
        cl_ulong global_mem_size;
        cl_ulong max_mem_alloc_size;
        cl_ulong real_alloc_size;
        cl_mem mem;
        cl_int err;
        int could_allocate;
        int could_write;
        int could_read;
        double alloc_time;
        double warmup_time;
        double write_time;
        double read_time;

        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_NAME, 256, name, NULL));
        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof (cl_ulong), &global_mem_size, NULL));
        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof (cl_ulong), &max_mem_alloc_size, NULL));

        context = clCreateContext (NULL, 1, &devices[i], NULL, NULL, &err);
        queue = clCreateCommandQueue (context, devices[i], CL_QUEUE_PROFILING_ENABLE, &err);

        OCL_CHECK_ERROR (err);

        real_alloc_size = max_mem_alloc_size;
        g_timer_start (timer);
        mem = clCreateBuffer (context, CL_MEM_READ_WRITE, real_alloc_size, NULL, &err);
        g_timer_stop (timer);
        alloc_time = g_timer_elapsed (timer, NULL);
        could_allocate = err == CL_SUCCESS;

        if (could_allocate) {
            char *data;

            data = malloc (real_alloc_size);

            g_timer_start (timer);
            err = clEnqueueWriteBuffer (queue, mem, CL_TRUE, 0, real_alloc_size, data, 0, NULL, NULL);
            g_timer_stop (timer);
            warmup_time = g_timer_elapsed (timer, NULL);

            g_timer_start (timer);
            err = clEnqueueWriteBuffer (queue, mem, CL_TRUE, 0, real_alloc_size, data, 0, NULL, NULL);
            g_timer_stop (timer);
            write_time = g_timer_elapsed (timer, NULL);
            could_write = err == CL_SUCCESS;

            g_timer_start (timer);
            err = clEnqueueReadBuffer (queue, mem, CL_TRUE, 0, real_alloc_size, data, 0, NULL, NULL);
            g_timer_stop (timer);
            read_time = g_timer_elapsed (timer, NULL);
            could_read = err == CL_SUCCESS;

            free (data);
        }

        g_print ("%s\n"
                 "  CL_DEVICE_GLOBAL_MEM_SIZE    : %-11lu B (%3.2f MB)\n"
                 "  CL_DEVICE_MAX_MEM_ALLOC_SIZE : %-11lu B (%3.2f MB, %3.1f%%)\n"
                 "  Could allocate               : %s (%3.5f s)\n"
                 "  Could write                  : %s (%3.5f s, %3.2f MB/s, warm up: %3.5fs)\n"
                 "  Could read                   : %s (%3.5f s, %3.2f MB/s)\n",
                 name,
                 global_mem_size, global_mem_size / 1024. / 1024.,
                 max_mem_alloc_size, max_mem_alloc_size / 1024. / 1024.,
                 ((double) max_mem_alloc_size) / global_mem_size * 100,
                 bools[could_allocate], alloc_time,
                 bools[could_write], write_time, real_alloc_size / write_time / 1024. / 1024., warmup_time,
                 bools[could_read], read_time, real_alloc_size / read_time / 1024. / 1024.);

        OCL_CHECK_ERROR (clReleaseCommandQueue (queue));
        OCL_CHECK_ERROR (clReleaseContext (context));

        if (i < num_devices - 1)
            g_print ("\n");
    }

    ocl_free (ocl);
    g_timer_destroy (timer);
}
