#include <stdio.h>
#include <stdlib.h>
#include "ocl.h"

char *
make_kernel (cl_ulong local_size)
{
    char *source;
    const size_t MAX_SOURCE_SIZE = 4096;
    char template[] = 
"__kernel void foo(__global char *in)\n\
{\n\
    __local char shared[%ld];\n\
    int idx = get_global_id(0);\n\
    if (idx < %ld)\n\
        shared[idx] = in[idx];\n\
}\0";

    source = malloc (MAX_SOURCE_SIZE);
    snprintf (source, MAX_SOURCE_SIZE, template, local_size, local_size);
    return source;
}

int
main (void)
{
    OclPlatform *ocl;
    cl_mem mem;
    cl_program program;
    cl_kernel kernel;
    cl_int errcode;
    cl_event event;
    cl_command_queue *cmd_queues;
    cl_ulong max_local_size;
    char *source;
    size_t n_elements;

    ocl = ocl_new (CL_DEVICE_TYPE_ALL, 1);

    if (ocl == NULL)
        return 1;

    OCL_CHECK_ERROR (clGetDeviceInfo (ocl_get_devices (ocl)[0],
                                      CL_DEVICE_LOCAL_MEM_SIZE,
                                      sizeof (cl_ulong),
                                      &max_local_size,
                                      NULL));

    printf ("Local mem size: %d bytes\n", (unsigned) max_local_size);

    source = make_kernel (max_local_size);
    program = ocl_create_program_from_source (ocl, source, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);
    free (source);

    cmd_queues = ocl_get_cmd_queues (ocl);
    kernel = clCreateKernel (program, "foo", &errcode);
    OCL_CHECK_ERROR (errcode);

    n_elements = max_local_size * 2;
    mem = clCreateBuffer (ocl_get_context (ocl), CL_MEM_READ_WRITE,
                          n_elements,
                          NULL, &errcode);

    OCL_CHECK_ERROR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &mem));
    OCL_CHECK_ERROR (clEnqueueNDRangeKernel (cmd_queues[0], kernel,
                                             1, NULL, &n_elements, NULL,
                                             0, NULL, &event));
                                             
    OCL_CHECK_ERROR (clWaitForEvents (1, &event));

    OCL_CHECK_ERROR (clReleaseEvent (event));
    OCL_CHECK_ERROR (clReleaseMemObject (mem));
    OCL_CHECK_ERROR (clReleaseKernel (kernel));
    OCL_CHECK_ERROR (clReleaseProgram (program));

    ocl_free (ocl);

    return 0;
}
