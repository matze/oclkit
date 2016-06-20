#include <glib.h>
#include <stdio.h>
#include <ocl.h>

typedef struct {
    unsigned long min;
    unsigned long max;
    double mean;
} Stats;


static const char* source =
    "__kernel void touch(void) "
    "{ "
    "   1 + 1; "
    "} ";


static void
get_opencl_event_times (cl_event event, unsigned long *start_wait, unsigned long *execution)
{
    cl_ulong queued, start, end;

    OCL_CHECK_ERROR (clGetEventProfilingInfo (event, CL_PROFILING_COMMAND_QUEUED, sizeof (cl_ulong), &queued, NULL));
    OCL_CHECK_ERROR (clGetEventProfilingInfo (event, CL_PROFILING_COMMAND_START, sizeof (cl_ulong), &start, NULL));
    OCL_CHECK_ERROR (clGetEventProfilingInfo (event, CL_PROFILING_COMMAND_END, sizeof (cl_ulong), &end, NULL));

    *start_wait = start - queued;
    *execution = end - start;
}

static void
get_event_times (int num_events, cl_event *events, Stats *wait, Stats *exec)
{
    wait->min = exec->min = G_MAXULONG;
    wait->max = exec->max = 0;
    wait->mean = exec->mean = 0.0;

    for (int i = 0; i < num_events; i++) {
        unsigned long start_wait;
        unsigned long execution;

        get_opencl_event_times (events[i], &start_wait, &execution);

        wait->mean += start_wait;
        exec->mean += execution;

        wait->min = MIN (wait->min, start_wait);
        wait->max = MAX (wait->max, start_wait);
        exec->min = MIN (exec->min, execution);
        exec->max = MAX (exec->max, execution);
    }

    wait->mean /= num_events;
    exec->mean /= num_events;
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
    cl_event *events;
    const int NUM_RUNS = 100;

    ocl = ocl_new_from_args (argc, argv, CL_QUEUE_PROFILING_ENABLE);

    program = ocl_create_program_from_source (ocl, source, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    kernel = clCreateKernel (program, "touch", &errcode);
    OCL_CHECK_ERROR (errcode);

    num_devices = ocl_get_num_devices (ocl);
    devices = ocl_get_devices (ocl);
    queues = ocl_get_cmd_queues (ocl);
    timer = g_timer_new ();
    events = g_new0 (cl_event, NUM_RUNS);

    for (int i = 0; i < num_devices; i++) {
        char name[256];
        cl_event event;
        size_t size = 16;
        const int NUM_WARMUP = 10;
        double wall_clock = 0.0;
        Stats wait;
        Stats exec;

        for (int r = 0; r < NUM_WARMUP; r++) {
            OCL_CHECK_ERROR (clEnqueueNDRangeKernel (queues[i], kernel, 1, NULL, &size, NULL, 0, NULL, &event));
            OCL_CHECK_ERROR (clWaitForEvents (1, &event));
            OCL_CHECK_ERROR (clReleaseEvent (event));
        }

        g_timer_start (timer);

        for (int r = 0; r < NUM_RUNS; r++) {
            OCL_CHECK_ERROR (clEnqueueNDRangeKernel (queues[i], kernel, 
                                                     1, NULL, &size, NULL,
                                                     r == 0 ? 0 : 1, r == 0 ? NULL : &events[r-1], &events[r]));
        }

        OCL_CHECK_ERROR (clWaitForEvents (1, &events[NUM_RUNS - 1]));

        g_timer_stop (timer);
        wall_clock = g_timer_elapsed (timer, NULL);

        get_event_times (NUM_RUNS, events, &wait, &exec);

        for (int r = 0; r < NUM_RUNS; r++)
            OCL_CHECK_ERROR (clReleaseEvent (events[r]));

        OCL_CHECK_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_NAME, 256, name, NULL));

        g_print ("%s\n"
                 "  wait for start: %8.5f us [min=%3.4f, max=%3.4f]\n"
                 "  execution time: %8.5f us [min=%3.4f, max=%3.4f]\n"
                 "  wall clock    : %8.5f us\n",
                 name,
                 wait.mean / 1000, wait.min / 1000.0, wait.max / 1000.0,
                 exec.mean / 1000, exec.min / 1000.0, exec.max / 1000.0,
                 wall_clock / NUM_RUNS * 1000 * 1000);

        if (i < num_devices - 1)
            g_print ("\n");
    }

    g_timer_destroy (timer);
    clReleaseKernel (kernel);
    clReleaseProgram (program);

    ocl_free (ocl);
}
