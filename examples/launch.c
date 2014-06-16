#include <glib.h>
#include <stdio.h>
#include <ocl.h>


static const char* source =
    "__kernel void touch(void) "
    "{ "
    "   1 + 1; "
    "} ";


static void
get_event_times (cl_event event, unsigned long *submission_wait, unsigned long *execution)
{
    cl_ulong queued, submitted, start, end;

    OCL_CHECK_ERROR (clGetEventProfilingInfo (event, CL_PROFILING_COMMAND_QUEUED, sizeof (cl_ulong), &queued, NULL));
    OCL_CHECK_ERROR (clGetEventProfilingInfo (event, CL_PROFILING_COMMAND_SUBMIT, sizeof (cl_ulong), &submitted, NULL));
    OCL_CHECK_ERROR (clGetEventProfilingInfo (event, CL_PROFILING_COMMAND_START, sizeof (cl_ulong), &start, NULL));
    OCL_CHECK_ERROR (clGetEventProfilingInfo (event, CL_PROFILING_COMMAND_END, sizeof (cl_ulong), &end, NULL));

    *submission_wait = submitted - queued;
    *execution = end - start;
}

int
main (int argc, const char **argv)
{
    OclPlatform *ocl;
    cl_program program;
    cl_device_id *devices;
    cl_command_queue *queues;
    cl_kernel kernel;
    cl_int errcode;
    int num_devices;
    GTimer *timer;

    ocl = ocl_new_from_args (argc, argv, CL_QUEUE_PROFILING_ENABLE);

    program = ocl_create_program_from_source (ocl, source, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    kernel = clCreateKernel (program, "touch", &errcode);
    OCL_CHECK_ERROR (errcode);

    num_devices = ocl_get_num_devices (ocl);
    devices = ocl_get_devices (ocl);
    queues = ocl_get_cmd_queues (ocl);
    timer = g_timer_new ();

    for (int i = 0; i < num_devices; i++) {
        char name[256];
        cl_event event;
        size_t size = 16;
        const int NUM_RUNS = 50000;
        unsigned long total_wait = 0;
        unsigned long total_execution = 0;
        double wall_clock = 0.0;

        for (int r = 0; r < NUM_RUNS; r++) {
            unsigned long wait;
            unsigned long execution;

            g_timer_start (timer);
            OCL_CHECK_ERROR (clEnqueueNDRangeKernel (queues[i], kernel, 
                        1, NULL, &size, NULL,
                        0, NULL, &event));

            clWaitForEvents (1, &event);
            g_timer_stop (timer);

            wall_clock += g_timer_elapsed (timer, NULL);

            get_event_times (event, &wait, &execution);
            clReleaseEvent (event);

            total_wait += wait;
            total_execution += execution;
        }

        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_NAME, 256, name, NULL));

        /* all times in nano seconds */
        printf ("%s %f %f %f\n", name,
                total_wait / ((double) NUM_RUNS),
                total_execution / ((double) NUM_RUNS),
                wall_clock / NUM_RUNS * 1000 * 1000 * 1000);
    }

    g_timer_destroy (timer);
    clReleaseKernel (kernel);
    clReleaseProgram (program);

    ocl_free (ocl);
}
