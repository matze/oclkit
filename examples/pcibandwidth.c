#include <glib.h>
#include <ocl.h>


typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    GTimer *upload;
    GTimer *download;
} App;


static const char* source =
    "__kernel void touch(global char *array) "
    "{ "
    "   array[0] = array[1] + array[2];"
    "} ";


void
measure_transfer_normal (App *app, size_t size)
{
    cl_int errcode;
    cl_mem buffer;
    char *array;

    buffer = clCreateBuffer (app->context, CL_MEM_READ_WRITE, size, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    array = g_malloc0 (size);

    g_timer_start (app->upload);
    OCL_CHECK_ERROR (clEnqueueWriteBuffer (app->queue, buffer, CL_TRUE, 0, size, array, 0, NULL, NULL));
    g_timer_stop (app->upload);

    /* Call a stupid kernel to avoid any optimizing assumptions of the OpenCL
     * run-time */

    OCL_CHECK_ERROR (clSetKernelArg (app->kernel, 0, sizeof (cl_mem), &buffer));
    OCL_CHECK_ERROR (clEnqueueNDRangeKernel (app->queue, app->kernel, 1, NULL, &size, NULL, 0, NULL, NULL));

    g_timer_start (app->download);
    OCL_CHECK_ERROR (clEnqueueReadBuffer (app->queue, buffer, CL_TRUE, 0, size, array, 0, NULL, NULL));
    g_timer_stop (app->download);

    g_free (array);
    clReleaseMemObject (buffer);
}

void
measure_transfer_pinned (App *app, size_t size)
{
    cl_int errcode;
    cl_mem buffer;
    cl_event event;
    char *array;

    buffer = clCreateBuffer (app->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    g_timer_start (app->upload);
    array = clEnqueueMapBuffer (app->queue, buffer, CL_TRUE, CL_MAP_WRITE, 0, size, 0, NULL, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    /* Should we do something here? */

    OCL_CHECK_ERROR (clEnqueueUnmapMemObject (app->queue, buffer, array, 0, NULL, NULL));
    g_timer_stop (app->upload);

    /* Call a stupid kernel to avoid any optimizing assumptions of the OpenCL
     * run-time */

    OCL_CHECK_ERROR (clSetKernelArg (app->kernel, 0, sizeof (cl_mem), &buffer));
    OCL_CHECK_ERROR (clEnqueueNDRangeKernel (app->queue, app->kernel, 1, NULL, &size, NULL, 0, NULL, &event));
    OCL_CHECK_ERROR (clWaitForEvents (1, &event));
    OCL_CHECK_ERROR (clReleaseEvent (event));

    g_timer_start (app->download);
    array = clEnqueueMapBuffer (app->queue, buffer, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    /* Should we do something here? */

    OCL_CHECK_ERROR (clEnqueueUnmapMemObject (app->queue, buffer, array, 0, NULL, NULL));
    g_timer_stop (app->download);

    clReleaseMemObject (buffer);
}


void run (App *app)
{
    const int NUM_RUNS = 3;

    g_print ("# size in bytes  /  upload MB/s  / download MB/s  /  upload/pinned MB/s  /  download/pinned MB/s\n");

#define NORMALIZE(x) (size / 1024. / 1024. / (x) / NUM_RUNS)

    for (size_t size = 4096; size < 256 * 1024 * 1024 + 1; size += 4096) {
        gdouble upload_normal = 0.0;
        gdouble download_normal = 0.0;
        gdouble upload_pinned = 0.0;
        gdouble download_pinned = 0.0;

        for (int i = 0; i < NUM_RUNS; i++) {
            measure_transfer_normal (app, size);

            upload_normal += g_timer_elapsed (app->upload, NULL);
            download_normal += g_timer_elapsed (app->download, NULL); 

            measure_transfer_pinned (app, size);
            upload_pinned += g_timer_elapsed (app->upload, NULL);
            download_pinned += g_timer_elapsed (app->download, NULL); 
        }

        g_print ("%zu  %.5f  %.5f  %.5f  %.5f\n", size,
                 NORMALIZE (upload_normal), NORMALIZE (download_normal),
                 NORMALIZE (upload_pinned), NORMALIZE (download_pinned));
    }

#undef NORMALIZE
}


int
main (void)
{
    OclPlatform *ocl;
    cl_int errcode;
    cl_program program;
    App app;

    ocl = ocl_new (CL_DEVICE_TYPE_GPU, 1);
    app.context = ocl_get_context (ocl);
    app.queue = ocl_get_cmd_queues (ocl)[0];

    program = ocl_create_program_from_source (ocl, source, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    app.kernel = clCreateKernel (program, "touch", &errcode);
    OCL_CHECK_ERROR (errcode);

    app.upload = g_timer_new ();
    app.download = g_timer_new ();

    run (&app);

    g_timer_destroy (app.upload);
    g_timer_destroy (app.download);

    clReleaseKernel (app.kernel);
    clReleaseProgram (program);

    ocl_free (ocl);
}
