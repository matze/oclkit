#include <glib.h>
#include <stdio.h>
#include <ocl.h>

int
main (int argc, const char **argv)
{
    OclPlatform *ocl;
    cl_device_id *devices;
    int num_devices;

    ocl = ocl_new_from_args (argc, argv, CL_QUEUE_PROFILING_ENABLE);

    num_devices = ocl_get_num_devices (ocl);
    devices = ocl_get_devices (ocl);

    for (int i = 0; i < num_devices; i++) {
        char name[256];
        cl_ulong global_mem_size;
        cl_ulong max_mem_alloc_size;

        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_NAME, 256, name, NULL));
        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof (cl_ulong), &global_mem_size, NULL));
        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof (cl_ulong), &max_mem_alloc_size, NULL));

        g_print ("%s\n"
                 "  CL_DEVICE_GLOBAL_MEM_SIZE    %12lu B (%3.2f MB)\n"
                 "  CL_DEVICE_MAX_MEM_ALLOC_SIZE %12lu B (%3.2f MB)\n",
                 name,
                 global_mem_size, global_mem_size / 1024. / 1024.,
                 max_mem_alloc_size, max_mem_alloc_size / 1024. / 1024.);

        if (i < num_devices - 1)
            g_print ("\n");
    }

    ocl_free (ocl);
}
