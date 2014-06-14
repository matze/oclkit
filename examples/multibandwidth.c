#include <glib.h>
#include <ocl.h>
#include <math.h>


typedef struct {
    OclPlatform *ocl;
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

static void
release_events (guint n_events, cl_event *events)
{
    for (guint i = 0; i < n_events; i++)
        OCL_CHECK_ERROR (clReleaseEvent (events[i]));
}

void
measure_transfer (App *app, GList *queue_list, size_t size, gdouble *upload, gdouble *download)
{
    cl_int errcode;
    cl_command_queue *queues;
    cl_mem *buffers;
    size_t buffer_size;
    cl_event *events;
    guint n_devices;
    char *array;
    GTimer *timer;
    
    array = g_malloc0 (size);
    n_devices = g_list_length (queue_list);
    queues = g_malloc0 (n_devices * sizeof (cl_command_queue));
    events = g_malloc0 (n_devices * sizeof (cl_event));
    buffers = g_malloc0 (n_devices * sizeof (cl_mem));
    buffer_size = size / n_devices;

    for (guint i = 0; i < n_devices; i++) {
        queues[i] = (cl_command_queue) g_list_nth_data (queue_list, i);
        buffers[i] = clCreateBuffer (app->context, CL_MEM_READ_WRITE, buffer_size, NULL, &errcode);
        OCL_CHECK_ERROR (errcode);
    }

    timer = g_timer_new ();

    for (guint i = 0; i < app->num_runs; i++) {
        /* Write data to device */
        g_timer_start (timer);

        for (guint i = 0; i < n_devices; i++) {
            OCL_CHECK_ERROR (clEnqueueWriteBuffer (queues[i], buffers[i], CL_TRUE, 0, buffer_size, array + i * buffer_size, 0, NULL, &events[i]));
        }

        OCL_CHECK_ERROR (clWaitForEvents (n_devices, events));
        g_timer_stop (timer);

        release_events (n_devices, events);

        *upload += g_timer_elapsed (timer, NULL);

        /* Call a stupid kernel to avoid any optimizing assumptions of the OpenCL
         * run-time */

        for (guint i = 0; i < n_devices; i++) {
            OCL_CHECK_ERROR (clSetKernelArg (app->kernel, 0, sizeof (cl_mem), &buffers[i]));
            OCL_CHECK_ERROR (clEnqueueNDRangeKernel (queues[i], app->kernel, 1, NULL, &buffer_size, NULL, 0, NULL, &events[i]));
        }

        OCL_CHECK_ERROR (clWaitForEvents (n_devices, events));
        release_events (n_devices, events);

        /* Read data from device to host */
        g_timer_start (timer);

        for (guint i = 0; i < n_devices; i++) {
            OCL_CHECK_ERROR (clEnqueueReadBuffer (queues[i], buffers[i], CL_FALSE, 0, buffer_size, array + i * buffer_size, 0, NULL, &events[i]));
        }

        OCL_CHECK_ERROR (clWaitForEvents (n_devices, events));
        g_timer_stop (timer);

        release_events (n_devices, events);

        *download += g_timer_elapsed (timer, NULL);
    }

    for (guint i = 0; i < n_devices; i++)
        OCL_CHECK_ERROR (clReleaseMemObject (buffers[i]));

    g_free (queues);
    g_free (events);
    g_free (buffers);
    g_free (array);
    g_timer_destroy (timer);
}

static gboolean
update_characteristic_vector (guint *cvec, guint n_length)
{
    /*
     * Determines a power set step according to the algorithm from
     * http://www.martinbroadhurst.com/combinatorial-algorithms.html#power-set.
     */
    gint ones = 0;

    for (guint i = 0; i < n_length; i++) {
        if (cvec[i] == 1) {
            if (i < n_length - 1 && cvec[i+1] == 0) {
                cvec[i] = 0;
                cvec[i+1] = 1;
                return TRUE;
            }

            ones++; 
        }
    }

    if (((guint) ones) == n_length)
        return FALSE;

    ones++;

    for (gint i = 0; i < ((gint) n_length); i++)
        cvec[i] = ones-- > 0 ? 1 : 0;

    return TRUE;
}

void run (App *app)
{
    cl_command_queue *queues;
    GString *device_id;
    guint *cvec;    /* characteristic vector to compute the power set */
    guint n_devices;
    guint n_sets;

    device_id = g_string_new (NULL);
    queues = ocl_get_cmd_queues (app->ocl);
    n_devices = ocl_get_num_devices (app->ocl);
    n_sets = pow (2, n_devices);
    cvec = g_malloc0 (n_sets);

    g_print ("# device(s)  /  size in bytes  /  upload MB/s  / download MB/s\n");

#define NORMALIZE(x) (size / 1024. / 1024. / ((x) / app->num_runs))

    while (update_characteristic_vector (cvec, n_devices)) {
        GList *queue_list = NULL;

        /* Build power set */
        g_string_erase (device_id, 0, -1);

        for (guint i = 0; i < n_devices; i++) {
            if (cvec[i]) {
                queue_list = g_list_append (queue_list, queues[i]);
                g_string_append_printf (device_id, "%i", i);
            }
        }

        for (size_t size = 256 * 1024; size < 128 * 1024 * 1024 + 1; size *= 2) {
            gdouble upload = 0.0;
            gdouble download = 0.0;

            measure_transfer (app, queue_list, size, &upload, &download);

            g_print ("%s  %zu  %.5f  %.5f\n", device_id->str, size,
                     NORMALIZE (upload), NORMALIZE (download));
        }

        g_list_free (queue_list);
    }

    g_free (cvec);
    g_string_free (device_id, TRUE);

#undef NORMALIZE
}


int
main (void)
{
    cl_int errcode;
    cl_program program;
    App app;

    app.ocl = ocl_new_with_queues (0, CL_DEVICE_TYPE_GPU, 0);
    app.context = ocl_get_context (app.ocl);

    program = ocl_create_program_from_source (app.ocl, source, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    app.kernel = clCreateKernel (program, "touch", &errcode);
    OCL_CHECK_ERROR (errcode);

    app.num_runs = 10;

    run (&app);

    clReleaseKernel (app.kernel);
    clReleaseProgram (program);

    ocl_free (app.ocl);
}
