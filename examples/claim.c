#include <stdio.h>
#include "ocl.h"

int
main (void)
{
    OclPlatform *ocl;
    cl_mem mem;
    cl_program program;
    cl_kernel kernel;
    cl_int errcode;
    cl_event event;
    size_t n_elements;

    cl_command_queue *cmd_queues;

    ocl = ocl_new_with_queues (0, CL_DEVICE_TYPE_ALL, 0);

    if (ocl == NULL)
        return 1;

    program = ocl_create_program_from_file (ocl, "test.cl", NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    cmd_queues = ocl_get_cmd_queues (ocl);
    kernel = clCreateKernel (program, "fill_ones", &errcode);
    OCL_CHECK_ERROR (errcode);

    n_elements = 1024 * 1024;
    mem = clCreateBuffer (ocl_get_context (ocl), CL_MEM_READ_WRITE,
                          n_elements * sizeof (float),
                          NULL, &errcode);

    OCL_CHECK_ERROR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    OCL_CHECK_ERROR (clEnqueueNDRangeKernel (cmd_queues[0], kernel,
                                             1, NULL, &n_elements, NULL,
                                             0, NULL, &event));
                                             
    OCL_CHECK_ERROR (clWaitForEvents (1, &event));
    OCL_CHECK_ERROR (clReleaseEvent (event));
    OCL_CHECK_ERROR (clReleaseMemObject (mem));

    fflush (stdin);
    printf ("Press Enter to continue ...\n");
    getchar ();

    OCL_CHECK_ERROR (clReleaseKernel (kernel));
    OCL_CHECK_ERROR (clReleaseProgram (program));

    ocl_free (ocl);

    fflush (stdin);
    printf ("Press Enter to exit ...\n");
    getchar ();

    return 0;
}
