#include <stdint.h>
#include <ocl.h>


static const char* source =
    "__kernel void test(global int *flags)"
    "{\n"
    "    flags[0] = 0;\n"
    "#if defined(cl_khr_fp64)\n"
    "    flags[0] |= 1 << 0;\n"
    "#endif\n"
    "#if defined(cl_amd_fp64)\n"
    "    flags[0] |= 1 << 1;\n"
    "#endif\n"
    "}";

int
main (int argc, const char **argv)
{
    OclPlatform *ocl;
    cl_context context;
    cl_device_type type;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int errcode;
    cl_mem buffer;
    int flags;
    int num_devices;
    size_t work_size = 1;
    unsigned int platform;

    platform = 0;
    type = CL_DEVICE_TYPE_GPU;

    if (ocl_read_args (argc, argv, &platform, &type))
        return 1;

    ocl = ocl_new (platform, type);
    context = ocl_get_context (ocl);
    num_devices = ocl_get_num_devices (ocl);

    for (int i = 0; i < num_devices; i++) {
        static char name[256];

        device = ocl_get_devices (ocl)[i];
        OCL_CHECK_ERROR (clGetDeviceInfo (device, CL_DEVICE_NAME, 256, name, NULL));

        queue = clCreateCommandQueue (context, device, 0, &errcode);
        OCL_CHECK_ERROR (errcode);

        program = ocl_create_program_from_source (ocl, source, NULL, &errcode);
        OCL_CHECK_ERROR (errcode);

        kernel = clCreateKernel (program, "test", &errcode);
        OCL_CHECK_ERROR (errcode);

        buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof (cl_int), NULL, &errcode);
        OCL_CHECK_ERROR (errcode);

        OCL_CHECK_ERROR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &buffer));
        OCL_CHECK_ERROR (clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &work_size, NULL, 0, NULL, NULL));
        OCL_CHECK_ERROR (clEnqueueReadBuffer (queue, buffer, CL_TRUE, 0, 1, &flags, 0, NULL, NULL));

        printf ("%s\n"
                "  cl_khr_fp64 = %i\n"
                "  cl_amd_fp64 = %i\n", name, flags & (1 << 0), (flags & (1 << 1)) >> 1);

        if (i < num_devices - 1)
            printf ("\n");

        OCL_CHECK_ERROR (clReleaseMemObject (buffer));
        OCL_CHECK_ERROR (clReleaseKernel (kernel));
        OCL_CHECK_ERROR (clReleaseProgram (program));
    }

    ocl_free (ocl);
}
