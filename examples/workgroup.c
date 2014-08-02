#include <stdio.h>
#include <ocl.h>


static const char* source =
    "__kernel void dim_1(global unsigned int *size)"
    "{ "
    "   if (get_global_id (0) == 0) { size[0] = get_local_size(0); size[1] = 0; }"
    "} "
    "__kernel void dim_2(global unsigned int *size)"
    "{ "
    "   if (get_global_id (0) == 0 && get_global_id (1) == 0) {size[0] = get_local_size(0); size[1] = get_local_size(1);}"
    "} ";


int
main (int argc, const char **argv)
{
    OclPlatform *ocl;
    cl_program program;
    cl_device_id *devices;
    cl_command_queue *queues;
    cl_kernel kernels[2];
    cl_mem buffer;
    size_t buffer_size;
    cl_int errcode;
    int num_devices;

    ocl = ocl_new_from_args (argc, argv, CL_QUEUE_PROFILING_ENABLE);

    program = ocl_create_program_from_source (ocl, source, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    kernels[0] = clCreateKernel (program, "dim_1", &errcode);
    OCL_CHECK_ERROR (errcode);
    kernels[1] = clCreateKernel (program, "dim_2", &errcode);
    OCL_CHECK_ERROR (errcode);

    buffer_size = 4 * sizeof(unsigned int);
    buffer = clCreateBuffer (ocl_get_context (ocl), CL_MEM_READ_WRITE, 
                             buffer_size, NULL, &errcode);

    OCL_CHECK_ERROR (errcode);

    num_devices = ocl_get_num_devices (ocl);
    devices = ocl_get_devices (ocl);
    queues = ocl_get_cmd_queues (ocl);

    for (int i = 0; i < num_devices; i++) {
        cl_kernel kernel;
        size_t max_work_item_sizes[4];
        size_t size[2];
        size_t max = 4096;

        unsigned int local[4] = {0,0,0,0};

        kernel = kernels[1];

        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                          3 * sizeof (size_t), max_work_item_sizes, NULL));

        printf ("device %i -> %zu %zu %zu\n", i,
                max_work_item_sizes[0], max_work_item_sizes[1], max_work_item_sizes[2]);

        for (size_t size_x = 8; size_x < max; size_x += 8) {
            for (size_t size_y = 1; size_y < 1025; size_y *= 2) {
                size[0] = size_x;
                size[1] = size_y;

                OCL_CHECK_ERROR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &buffer));
                OCL_CHECK_ERROR (clEnqueueNDRangeKernel (queues[i], kernel,
                                                         2, NULL, size, NULL,
                                                         0, NULL, NULL));
                OCL_CHECK_ERROR (clEnqueueReadBuffer (queues[i], buffer, CL_TRUE, 
                                                      0, buffer_size, local, 0, NULL, NULL));

                printf ("%i %4zu %4zu %4u %4u\n", i, size_x, size_y, local[0], local[1]);
            }
        }

        /* OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_NAME, 256, name, NULL)); */

        /* all times in nano seconds */
        /* printf ("%s %f %f %f\n", name, */
        /*         total_wait / ((double) NUM_RUNS), */
        /*         total_execution / ((double) NUM_RUNS), */
        /*         wall_clock / NUM_RUNS * 1000 * 1000 * 1000); */
    }

    for (int i = 0; i < 2; i++)
        clReleaseKernel (kernels[i]);

    clReleaseProgram (program);
    clReleaseMemObject (buffer);

    ocl_free (ocl);
}
