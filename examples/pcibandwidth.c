#include <glib.h>
#include <ocl.h>


typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    guint num_runs;
} App;

static const char* source =
    "__kernel void touch(global char *array) "
    "{ "
    "   array[0] = array[1] + array[2];"
    "} ";


void
measure_transfer_normal (App *app, size_t size, gdouble *upload, gdouble *download)
{
    cl_int errcode;
    cl_mem buffer;
    cl_event event;
    char *array;
    GTimer *timer;

    timer = g_timer_new ();
    buffer = clCreateBuffer (app->context, CL_MEM_READ_WRITE, size, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    array = g_malloc0 (size);

    for (guint i = 0; i < app->num_runs; i++) {
        g_timer_start (timer);
        OCL_CHECK_ERROR (clEnqueueWriteBuffer (app->queue, buffer, CL_TRUE, 0, size, array, 0, NULL, NULL));
        g_timer_stop (timer);

        *upload += g_timer_elapsed (timer, NULL);

        /* Call a stupid kernel to avoid any optimizing assumptions of the OpenCL
         * run-time */

        OCL_CHECK_ERROR (clSetKernelArg (app->kernel, 0, sizeof (cl_mem), &buffer));
        OCL_CHECK_ERROR (clEnqueueNDRangeKernel (app->queue, app->kernel, 1, NULL, &size, NULL, 0, NULL, &event));
        OCL_CHECK_ERROR (clWaitForEvents (1, &event));
        OCL_CHECK_ERROR (clReleaseEvent (event));

        g_timer_start (timer);
        OCL_CHECK_ERROR (clEnqueueReadBuffer (app->queue, buffer, CL_TRUE, 0, size, array, 0, NULL, NULL));
        g_timer_stop (timer);

        *download += g_timer_elapsed (timer, NULL);
    }

    g_free (array);
    clReleaseMemObject (buffer);
    g_timer_destroy (timer);
}

void
measure_transfer_pinned (App *app, size_t size, gdouble *upload, gdouble *download)
{
    cl_int errcode;
    cl_mem buffer;
    cl_event event;
    char *array;
    GTimer *timer;

    timer = g_timer_new ();
    buffer = clCreateBuffer (app->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    for (guint i = 0; i < app->num_runs; i++) {
        g_timer_start (timer);
        array = clEnqueueMapBuffer (app->queue, buffer, CL_TRUE, CL_MAP_WRITE, 0, size, 0, NULL, NULL, &errcode);
        OCL_CHECK_ERROR (errcode);

        /* Should we do something here? */

        OCL_CHECK_ERROR (clEnqueueUnmapMemObject (app->queue, buffer, array, 0, NULL, &event));
        OCL_CHECK_ERROR (clWaitForEvents (1, &event));
        OCL_CHECK_ERROR (clReleaseEvent (event));
        g_timer_stop (timer);

        *upload += g_timer_elapsed (timer, NULL);

        /* Call a stupid kernel to avoid any optimizing assumptions of the OpenCL
         * run-time */

        OCL_CHECK_ERROR (clSetKernelArg (app->kernel, 0, sizeof (cl_mem), &buffer));
        OCL_CHECK_ERROR (clEnqueueNDRangeKernel (app->queue, app->kernel, 1, NULL, &size, NULL, 0, NULL, &event));
        OCL_CHECK_ERROR (clWaitForEvents (1, &event));
        OCL_CHECK_ERROR (clReleaseEvent (event));

        g_timer_start (timer);
        array = clEnqueueMapBuffer (app->queue, buffer, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &errcode);
        OCL_CHECK_ERROR (errcode);

        /* Should we do something here? */

        OCL_CHECK_ERROR (clEnqueueUnmapMemObject (app->queue, buffer, array, 0, NULL, &event));
        OCL_CHECK_ERROR (clWaitForEvents (1, &event));
        OCL_CHECK_ERROR (clReleaseEvent (event));
        g_timer_stop (timer);
        *download += g_timer_elapsed (timer, NULL);
    }

    clReleaseMemObject (buffer);
    g_timer_destroy (timer);
}


void run (App *app)
{
    g_print ("# size in bytes  /  upload MB/s  / download MB/s  /  upload/pinned MB/s  /  download/pinned MB/s\n");

#define NORMALIZE(x) (size / 1024. / 1024. / ((x) / app->num_runs))

    for (size_t size = 256 * 1024; size < 64 * 1024 * 1024 + 1; size += 256 * 1024) {
        gdouble upload_normal = 0.0;
        gdouble download_normal = 0.0;
        gdouble upload_pinned = 0.0;
        gdouble download_pinned = 0.0;

        measure_transfer_normal (app, size, &upload_normal, &download_normal);
        measure_transfer_pinned (app, size, &upload_pinned, &download_pinned);

        g_print ("%zu  %.5f  %.5f  %.5f  %.5f\n", size,
                 NORMALIZE (upload_normal), NORMALIZE (download_normal),
                 NORMALIZE (upload_pinned), NORMALIZE (download_pinned));
    }

#undef NORMALIZE
}


int
main (int argc, const char **argv)
{
    OclPlatform *ocl;
    cl_int errcode;
    cl_program program;
    gchar device_name[256];
    App app;

    ocl = ocl_new_from_args (argc, argv, 0);
    app.context = ocl_get_context (ocl);
    app.queue = ocl_get_cmd_queues (ocl)[0];

    clGetDeviceInfo (ocl_get_devices (ocl)[0], CL_DEVICE_NAME, 256, device_name, NULL);
    g_print ("# running on %s\n", device_name);

    program = ocl_create_program_from_source (ocl, source, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    app.kernel = clCreateKernel (program, "touch", &errcode);
    OCL_CHECK_ERROR (errcode);

    app.num_runs = 3;

    run (&app);

    clReleaseKernel (app.kernel);
    clReleaseProgram (program);

    ocl_free (ocl);
}
