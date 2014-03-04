#include <glib.h>
#include <math.h>
#include "ocl.h"


int
main (void)
{
    OclPlatform *ocl;
    cl_context context;
    cl_int errcode;
    float *host_sin_lut, *host_cos_lut;
    float *host_lut_result, *host_comp_result;
    cl_mem dev_sin_lut, dev_cos_lut;
    cl_mem dev_lut_result, dev_comp_result;
    cl_program program;
    cl_kernel lut_kernel, comp_kernel;
    cl_event event;
    cl_command_queue queue;
    GTimer *timer;

    const int N_ITERATIONS = 256;
    const size_t N_ELEMENTS = 1024;
    const size_t N_ELEMENTS_2 = N_ELEMENTS * N_ELEMENTS;
    const size_t SIZE = N_ELEMENTS * sizeof(float);
    const size_t SIZE_2 = N_ELEMENTS_2 * sizeof(float);

    size_t work_size[2] = { N_ELEMENTS, N_ELEMENTS };

    ocl = ocl_new (CL_DEVICE_TYPE_ALL, 1);

    if (ocl == NULL)
        return 1;

    host_sin_lut = g_malloc0 (SIZE);
    host_cos_lut = g_malloc0 (SIZE);
    host_lut_result = g_malloc0 (SIZE_2);
    host_comp_result = g_malloc0 (SIZE_2);

    for (size_t i = 0; i < N_ELEMENTS; i++) {
        host_sin_lut[i] = sin (i);
        host_cos_lut[i] = cos (i);
    }

    context = ocl_get_context (ocl);

    /* Setup buffers */
    dev_sin_lut = clCreateBuffer (context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  SIZE, host_sin_lut, &errcode);
    OCL_CHECK_ERROR (errcode);

    dev_cos_lut = clCreateBuffer (context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  SIZE, host_cos_lut, &errcode);
    OCL_CHECK_ERROR (errcode);

    dev_lut_result = clCreateBuffer (context, CL_MEM_WRITE_ONLY,
                                     SIZE_2, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    dev_comp_result = clCreateBuffer (context, CL_MEM_WRITE_ONLY,
                                      SIZE_2, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);


    /* Setup kernels */
    program = ocl_create_program_from_file (ocl, "lut.cl", NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    lut_kernel = clCreateKernel (program, "fill_with_lut", &errcode);
    OCL_CHECK_ERROR (errcode);

    comp_kernel = clCreateKernel (program, "fill_with_comp", &errcode);
    OCL_CHECK_ERROR (errcode);

    OCL_CHECK_ERROR (clSetKernelArg (lut_kernel, 0, sizeof (cl_mem), &dev_sin_lut));
    OCL_CHECK_ERROR (clSetKernelArg (lut_kernel, 1, sizeof (cl_mem), &dev_cos_lut));
    OCL_CHECK_ERROR (clSetKernelArg (lut_kernel, 2, sizeof (cl_mem), &dev_lut_result));
    
    OCL_CHECK_ERROR (clSetKernelArg (comp_kernel, 0, sizeof (cl_mem), &dev_comp_result));


    /* Compute */
    timer = g_timer_new (); 
    queue = ocl_get_cmd_queues (ocl)[0];
    g_timer_start (timer);

    for (int i = 0; i < N_ITERATIONS; i++) {
        OCL_CHECK_ERROR (clEnqueueNDRangeKernel (queue, lut_kernel,
                                                 2, NULL, work_size, NULL,
                                                 0, NULL, &event));

        OCL_CHECK_ERROR (clWaitForEvents (1, &event));
        OCL_CHECK_ERROR (clReleaseEvent (event));
    }

    g_timer_stop (timer);
    g_print ("LUT: %fs\n", g_timer_elapsed (timer, NULL));

    g_timer_start (timer);

    for (int i = 0; i < N_ITERATIONS; i++) {
        OCL_CHECK_ERROR (clEnqueueNDRangeKernel (queue, comp_kernel,
                                                 2, NULL, work_size, NULL,
                                                 0, NULL, &event));

        OCL_CHECK_ERROR (clWaitForEvents (1, &event));
        OCL_CHECK_ERROR (clReleaseEvent (event));
    }

    g_timer_stop (timer);
    g_print ("Comp: %fs\n", g_timer_elapsed (timer, NULL));

    /* Clean up */
    OCL_CHECK_ERROR (clReleaseKernel (lut_kernel));
    OCL_CHECK_ERROR (clReleaseKernel (comp_kernel));
    OCL_CHECK_ERROR (clReleaseProgram (program));
    OCL_CHECK_ERROR (clReleaseMemObject (dev_lut_result));
    OCL_CHECK_ERROR (clReleaseMemObject (dev_comp_result));
    OCL_CHECK_ERROR (clReleaseMemObject (dev_sin_lut));
    OCL_CHECK_ERROR (clReleaseMemObject (dev_cos_lut));

    ocl_free (ocl);

    g_free (host_lut_result);
    g_free (host_comp_result);
    g_free (host_sin_lut);
    g_free (host_cos_lut);
    return 0;
}
