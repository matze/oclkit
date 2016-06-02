#include <ocl.h>
#include <stdio.h>

int
main (int argc, const char **argv)
{
    OclPlatform *ocl;
    cl_device_id *devices;

    ocl = ocl_new_from_args (argc, argv, 0);
    devices = ocl_get_devices (ocl);

    for (int i = 0; i < ocl_get_num_devices (ocl); i++) {
        static char name[256];
        size_t resolution;

        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_NAME, 256, name, NULL));
        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(size_t), &resolution, NULL));
        printf ("%-30s: %zu ns\n", name, resolution);
    }

    ocl_free (ocl);
}
