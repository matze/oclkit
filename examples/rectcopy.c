#include <glib.h>
#include <ocl.h>
#include <math.h>


typedef struct {
    OclPlatform *ocl;
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    guint num_runs;
    GTimer *timer;
} App;


static double
use_api (App *app, float *src, cl_mem dst,
         size_t src_origin[3], size_t src_size[3], size_t dst_size[3])
{
    size_t total;
    size_t dst_origin[] = {0, 0, 0};
    cl_mem large_mem;
    cl_int errcode;
    cl_event event;
    size_t src_row_pitch;
    size_t dst_row_pitch;

    g_timer_start (app->timer);

    src_row_pitch = sizeof (float) * src_size[1];
    dst_row_pitch = sizeof (float) * dst_size[1];
    total = sizeof (float) * src_size[0] * src_size[1] * src_size[2];
    large_mem = clCreateBuffer (app->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, total, src, &errcode);
    OCL_CHECK_ERROR (errcode);

    OCL_CHECK_ERROR (clEnqueueCopyBufferRect (app->queue, large_mem, dst,
                                              src_origin, dst_origin, dst_size,
                                              src_row_pitch, 0,
                                              dst_row_pitch, 0,
                                              0, NULL, &event));

    OCL_CHECK_ERROR (clWaitForEvents (1, &event));
    OCL_CHECK_ERROR (clReleaseEvent (event));
    OCL_CHECK_ERROR (clReleaseMemObject (large_mem));

    return g_timer_elapsed (app->timer, NULL);
}

void run (App *app)
{
}

int
main (void)
{
    App app;

    app.ocl = ocl_new (CL_DEVICE_TYPE_GPU, 1);
    app.context = ocl_get_context (app.ocl);
    app.queue = ocl_get_cmd_queues (app.ocl)[0];
    app.timer = g_timer_new ();
    app.num_runs = 10;

    run (&app);

    ocl_free (app.ocl);
}
