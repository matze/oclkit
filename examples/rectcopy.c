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

static size_t
float_size(size_t size[3])
{
    return sizeof (float) * size[0] * size[1] * size[2];
}

static double
use_api (App *app, float *src, cl_mem dst,
         size_t src_origin[3], size_t src_size[3], size_t dst_size[3])
{
    size_t total;
    size_t dst_origin[] = {0, 0, 0};
    cl_mem intermediate;
    cl_int errcode;
    cl_event event;
    size_t src_row_pitch;
    size_t dst_row_pitch;

    g_timer_start (app->timer);

    src_row_pitch = sizeof (float) * src_size[0];
    dst_row_pitch = sizeof (float) * dst_size[0];
    total = float_size (src_size);
    intermediate = clCreateBuffer (app->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, total, src, &errcode);
    OCL_CHECK_ERROR (errcode);

    OCL_CHECK_ERROR (clEnqueueCopyBufferRect (app->queue, intermediate, dst,
                                              src_origin, dst_origin, dst_size,
                                              src_row_pitch, 0,
                                              dst_row_pitch, 0,
                                              0, NULL, &event));

    OCL_CHECK_ERROR (clWaitForEvents (1, &event));
    OCL_CHECK_ERROR (clReleaseEvent (event));
    OCL_CHECK_ERROR (clReleaseMemObject (intermediate));
    g_timer_stop (app->timer);

    return g_timer_elapsed (app->timer, NULL);
}

static double
use_preassemble (App *app, float *src, cl_mem dst,
                 size_t src_origin[3], size_t src_size[3], size_t dst_size[3])
{
    float *intermediate;
    size_t dst_total;

    g_timer_start (app->timer);
    dst_total = float_size (dst_size);
    intermediate = malloc (dst_total);

    OCL_CHECK_ERROR (clEnqueueWriteBuffer (app->queue, dst, CL_TRUE, 0, dst_total, intermediate, 0, NULL, NULL));

    free (intermediate);
    g_timer_stop (app->timer);

    return g_timer_elapsed (app->timer, NULL);
}

void run (App *app)
{
    cl_int errcode;
    cl_mem dst;
    float *src;
    size_t src_size[3];
    size_t dst_size[3];
    size_t origin[3];
    double time;

    src_size[0] = 4096;
    src_size[1] = 4096;
    src_size[2] = 1;
    dst_size[0] = 1000;
    dst_size[1] = 500;
    dst_size[2] = 1;

    origin[0] = 0;
    origin[1] = 50;
    origin[2] = 0;

    src = malloc (float_size (src_size));
    dst = clCreateBuffer (app->context, CL_MEM_READ_ONLY, float_size (dst_size), NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    time = 0.0;

    for (guint i = 0; i < app->num_runs; i++)
        time += use_api (app, src, dst, origin, src_size, dst_size);

    printf ("API: %3.5f s\n", time / app->num_runs);

    time = 0.0;

    for (guint i = 0; i < app->num_runs; i++)
        time += use_preassemble (app, src, dst, origin, src_size, dst_size);

    printf ("pre-assemble: %3.5f s\n", time / app->num_runs);

    OCL_CHECK_ERROR (clReleaseMemObject (dst));
    free (src);
}

int
main (void)
{
    App app;

    app.ocl = ocl_new (CL_DEVICE_TYPE_ALL, 1);
    app.context = ocl_get_context (app.ocl);
    app.queue = ocl_get_cmd_queues (app.ocl)[0];
    app.timer = g_timer_new ();
    app.num_runs = 10;

    run (&app);

    ocl_free (app.ocl);
}
