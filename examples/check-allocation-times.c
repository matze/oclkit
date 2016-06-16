#include <glib.h>
#include <stdio.h>
#include <ocl.h>

static void
measure_allocation (cl_context context, cl_command_queue queue, size_t size, double *time)
{
    cl_mem mem;
    cl_int err;
    char *data;
    GTimer *timer;

    timer = g_timer_new ();
    mem = clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, &err);
    data = malloc (size);

    g_timer_start (timer);
    err = clEnqueueWriteBuffer (queue, mem, CL_TRUE, 0, size, data, 0, NULL, NULL);
    OCL_CHECK_ERROR (err);
    g_timer_stop (timer);
    *time = g_timer_elapsed (timer, NULL);

    free (data);
    OCL_CHECK_ERROR (clReleaseMemObject (mem));
    g_timer_destroy (timer);
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
        cl_ulong max_mem_alloc_size;
        cl_int err;

        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_NAME, 256, name, NULL));
        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof (cl_ulong), &max_mem_alloc_size, NULL));

        g_print ("%s\n", name);

        context = clCreateContext (NULL, 1, &devices[i], NULL, NULL, &err);
        queue = clCreateCommandQueue (context, devices[i], CL_QUEUE_PROFILING_ENABLE, &err);

        while (max_mem_alloc_size > 0) {
            double time;

            measure_allocation (context, queue, max_mem_alloc_size, &time);
            g_print ("  %-12zu %3.5f\n", max_mem_alloc_size, time);
            max_mem_alloc_size /= 2;
        }

        OCL_CHECK_ERROR (clReleaseCommandQueue (queue));
        OCL_CHECK_ERROR (clReleaseContext (context));

        if (i < num_devices - 1)
            g_print ("\n");
    }

    ocl_free (ocl);
}
