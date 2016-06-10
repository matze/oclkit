#include <glib.h>
#include <stdio.h>
#include <ocl.h>

int
main (int argc, const char **argv)
{
    OclPlatform *ocl;
    cl_context context;
    cl_device_id *devices;
    cl_command_queue *queues;
    int num_devices;
    static char *bools[] = {"no", "yes"};

    ocl = ocl_new_from_args (argc, argv, CL_QUEUE_PROFILING_ENABLE);

    context = ocl_get_context (ocl);
    num_devices = ocl_get_num_devices (ocl);
    devices = ocl_get_devices (ocl);
    queues = ocl_get_cmd_queues (ocl);

    for (int i = 0; i < num_devices; i++) {
        char name[256];
        cl_ulong global_mem_size;
        cl_ulong max_mem_alloc_size;
        cl_ulong real_alloc_size;
        cl_mem mem;
        cl_int err;
        int could_allocate;
        int could_write;

        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_NAME, 256, name, NULL));
        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof (cl_ulong), &global_mem_size, NULL));
        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof (cl_ulong), &max_mem_alloc_size, NULL));

        real_alloc_size = max_mem_alloc_size;
        mem = clCreateBuffer (context, CL_MEM_READ_WRITE, real_alloc_size, NULL, &err);
        could_allocate = err == CL_SUCCESS;

        if (could_allocate) {
            char *data;

            data = malloc (real_alloc_size);
            err = clEnqueueWriteBuffer (queues[i], mem, CL_TRUE, 0, real_alloc_size, data, 0, NULL, NULL);
            could_write = err == CL_SUCCESS;
            free (data);
        }

        g_print ("%s\n"
                 "  CL_DEVICE_GLOBAL_MEM_SIZE    : %-11lu B (%3.2f MB)\n"
                 "  CL_DEVICE_MAX_MEM_ALLOC_SIZE : %-11lu B (%3.2f MB, %3.1f%%)\n"
                 "  Could allocate               : %s\n"
                 "  Could write                  : %s\n",
                 name,
                 global_mem_size, global_mem_size / 1024. / 1024.,
                 max_mem_alloc_size, max_mem_alloc_size / 1024. / 1024.,
                 ((double) max_mem_alloc_size) / global_mem_size * 100,
                 bools[could_allocate], bools[could_write]);

        if (i < num_devices - 1)
            g_print ("\n");
    }

    ocl_free (ocl);
}
