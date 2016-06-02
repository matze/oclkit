#include <stdint.h>
#include <ocl.h>


typedef struct {
    cl_context context;
    cl_device_id device;
    cl_kernel kernel;
    cl_mem *buffers;
    size_t work_size;
    int n_times;
    char device_name[256];
} App;


static const char* source =
    "__kernel void compute(global int *data, int times)"
    "{ "
    "   size_t idx = get_global_id(0); "
    "   for (int i = 0; i < times; i++) "
    "       data[idx]++; "
    "} ";


static void
check_events (FILE *stream, int n_kernels, cl_event *events)
{
    cl_ulong *start;
    cl_ulong *end;
    cl_ulong earliest;

    earliest = UINT64_MAX;
    start = malloc (n_kernels * sizeof (cl_ulong));
    end  = malloc (n_kernels * sizeof (cl_ulong));

    for (int i = 0; i < n_kernels; i++) {
        ocl_get_event_times (events[i], &start[i], &end[i], NULL, NULL);
        OCL_CHECK_ERROR (clReleaseEvent (events[i]));

        if (start[i] < earliest)
            earliest = start[i];
    }

    for (int i = 0; i < n_kernels; i++) {
        fprintf (stream, " %lu %lu ", start[i] - earliest, end[i] - earliest);
    }

    free (start);
    free (end);
}

static void
measure_in_order_queue (App *app, FILE *stream, int n_kernels)
{
    cl_command_queue queue;
    cl_command_queue_properties props;
    cl_event sync_event;
    cl_event *events;
    cl_int errcode;

    /* Create out of order queue */
    props = CL_QUEUE_PROFILING_ENABLE;
    queue = clCreateCommandQueue (app->context, app->device, props, &errcode);
    OCL_CHECK_ERROR (errcode);

    events = malloc (n_kernels * sizeof(cl_event));
    sync_event = clCreateUserEvent (app->context, &errcode);

    for (int i = 0; i < n_kernels; i++) {
        OCL_CHECK_ERROR (clSetKernelArg (app->kernel, 0, sizeof (cl_mem), &app->buffers[i]));
        OCL_CHECK_ERROR (clSetKernelArg (app->kernel, 1, sizeof (int), &app->n_times));
        OCL_CHECK_ERROR (clEnqueueNDRangeKernel (queue, app->kernel,
                                                 1, NULL, &app->work_size, NULL,
                                                 1, &sync_event, &events[i]));
    }

    /* Start all kernels at once */
    OCL_CHECK_ERROR (clSetUserEventStatus (sync_event, CL_COMPLETE));

    /* Wait for completion */
    OCL_CHECK_ERROR (clWaitForEvents (n_kernels, events));

    check_events (stream, n_kernels, events);

    OCL_CHECK_ERROR (clReleaseEvent (sync_event));
    OCL_CHECK_ERROR (clReleaseCommandQueue (queue));

    free (events);
}

static void
measure_out_of_order_queue (App *app, FILE *stream, int n_kernels)
{
    cl_command_queue queue;
    cl_command_queue_properties props;
    cl_event sync_event;
    cl_event *events;
    cl_int errcode;

    /* Create out of order queue */
    props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;
    queue = clCreateCommandQueue (app->context, app->device, props, &errcode);
    OCL_CHECK_ERROR (errcode);

    events = malloc (n_kernels * sizeof(cl_event));
    sync_event = clCreateUserEvent (app->context, &errcode);

    for (int i = 0; i < n_kernels; i++) {
        OCL_CHECK_ERROR (clSetKernelArg (app->kernel, 0, sizeof (cl_mem), &app->buffers[i]));
        OCL_CHECK_ERROR (clSetKernelArg (app->kernel, 1, sizeof (int), &app->n_times));
        OCL_CHECK_ERROR (clEnqueueNDRangeKernel (queue, app->kernel,
                                                 1, NULL, &app->work_size, NULL,
                                                 1, &sync_event, &events[i]));
    }

    /* Start all kernels at once */
    OCL_CHECK_ERROR (clSetUserEventStatus (sync_event, CL_COMPLETE));

    /* Wait for completion */
    OCL_CHECK_ERROR (clWaitForEvents (n_kernels, events));

    check_events (stream, n_kernels, events);

    OCL_CHECK_ERROR (clReleaseEvent (sync_event));
    OCL_CHECK_ERROR (clReleaseCommandQueue (queue));

    free (events);
}

static void
measure_multi_queue (App *app, FILE *stream, int n_kernels)
{
    cl_command_queue *queues;
    cl_command_queue_properties props;
    cl_event sync_event;
    cl_event *events;
    cl_int errcode;

    /* Create out of order queue */
    queues = malloc (sizeof (cl_command_queue) * n_kernels);
    props = CL_QUEUE_PROFILING_ENABLE;

    events = malloc (n_kernels * sizeof(cl_event));
    sync_event = clCreateUserEvent (app->context, &errcode);

    for (int i = 0; i < n_kernels; i++) {
        queues[i] = clCreateCommandQueue (app->context, app->device, props, &errcode);
        OCL_CHECK_ERROR (errcode);

        OCL_CHECK_ERROR (clSetKernelArg (app->kernel, 0, sizeof (cl_mem), &app->buffers[i]));
        OCL_CHECK_ERROR (clSetKernelArg (app->kernel, 1, sizeof (int), &app->n_times));
        OCL_CHECK_ERROR (clEnqueueNDRangeKernel (queues[i], app->kernel,
                                                 1, NULL, &app->work_size, NULL,
                                                 1, &sync_event, &events[i]));
    }

    /* Start all kernels at once */
    OCL_CHECK_ERROR (clSetUserEventStatus (sync_event, CL_COMPLETE));

    /* Wait for completion */
    OCL_CHECK_ERROR (clWaitForEvents (n_kernels, events));

    check_events (stream, n_kernels, events);

    OCL_CHECK_ERROR (clReleaseEvent (sync_event));

    for (int i = 0; i < n_kernels; i++)
        OCL_CHECK_ERROR (clReleaseCommandQueue (queues[i]));

    free (queues);
    free (events);
}

static FILE *
open_file (App *app, const char *fname_fmt)
{
    char fname[256];

    snprintf (fname, 256, fname_fmt, app->device_name);
    return fopen(fname, "w");
}

static void
run (App *app)
{
    FILE *in_order_stream;
    FILE *out_of_order_stream;
    FILE *multi_stream;
    cl_int errcode;
    app->work_size = 1024;

    multi_stream = open_file (app, "multi-queue-%s.txt");
    in_order_stream = open_file (app, "in-order-queue-%s.txt");
    out_of_order_stream = open_file (app, "out-of-order-queue-%s.txt");

    for (int i = 2; i < 1024; i *= 2) {
        app->work_size = i;

        for (int n_kernels = 2; n_kernels < 8; n_kernels++) {
            app->buffers = malloc (n_kernels * sizeof (cl_mem));


            for (int i = 0; i < n_kernels; i++) {
                app->buffers[i] = clCreateBuffer (app->context, CL_MEM_READ_WRITE,
                                                  app->work_size, NULL, &errcode);
                OCL_CHECK_ERROR (errcode);
            }

            fprintf (out_of_order_stream, "\n%i %zu", n_kernels, app->work_size);
            measure_out_of_order_queue (app, out_of_order_stream, n_kernels);

            fprintf (in_order_stream, "\n%i %zu", n_kernels, app->work_size);
            measure_in_order_queue (app, in_order_stream, n_kernels);

            fprintf (multi_stream, "\n%i %zu", n_kernels, app->work_size);
            measure_multi_queue (app, multi_stream, n_kernels);

            for (int i = 0; i < n_kernels; i++) {
                OCL_CHECK_ERROR (clReleaseMemObject (app->buffers[i]));
            }

            free (app->buffers);
        }
    }

    fclose (in_order_stream);
    fclose (out_of_order_stream);
    fclose (multi_stream);
}

static void
sanitize (char *s, int n)
{
    for (int i = 0; i < n; i++) {
        if (s[i] == ' ')
            s[i] = '-';
    }
}

int
main (int argc, const char **argv)
{
    OclPlatform *ocl;
    unsigned int platform;
    cl_device_type type;
    cl_int errcode;
    cl_program program;
    App app;

    platform = 0;
    type = CL_DEVICE_TYPE_GPU;

    if (ocl_read_args (argc, argv, &platform, &type))
        return 1;

    ocl = ocl_new (platform, type);
    app.context = ocl_get_context (ocl);
    app.device = ocl_get_devices (ocl)[0];
    app.n_times = 1000;

    clGetDeviceInfo (ocl_get_devices (ocl)[0], CL_DEVICE_NAME, 256, app.device_name, NULL);
    printf ("# running on %s", app.device_name);

    sanitize (app.device_name, 256);

    program = ocl_create_program_from_source (ocl, source, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    app.kernel = clCreateKernel (program, "compute", &errcode);
    OCL_CHECK_ERROR (errcode);

    run (&app);

    OCL_CHECK_ERROR (clReleaseKernel (app.kernel));
    OCL_CHECK_ERROR (clReleaseProgram (program));

    ocl_free (ocl);
}
