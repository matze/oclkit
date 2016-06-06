#include <glib.h>
#include <glib-object.h>
#include <ocl.h>


typedef struct {
    GClosure *closure;
    size_t size;    /* total size of parameters */
    GType *types;
    gsize *type_sizes;
    guint n_params;
} Callback;


typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    gboolean stop;
    GHashTable *callbacks;
    GThread *listener;
    guint last;

    GString *aux_source;

    gsize param_size;
    gpointer param_host_data;
    cl_mem param_dev_data;
} App;


static void
handle_print (gfloat n, int a, int x)
{
    g_print ("output = %f, %i, %i \n", n, a, x);
}

static gchar *
get_param_list (GList *args)
{
    GString *str = g_string_new (NULL);
    guint i = 0;
    guint n = g_list_length (args);

    for (GList *it = g_list_first (args); it != NULL; it = g_list_next (it)) {
        const gchar *type = (const gchar *) it->data;

        g_string_append_printf (str, "%s param%i", type, i++);

        if (i < n)
            g_string_append (str, ", ");
    }

    return g_string_free (str, FALSE);
}

static gchar *
get_assignments (GList *args, gsize *sizes)
{
    GString *str = g_string_new (NULL);
    gsize current = 0;
    guint i = 0;

    for (GList *it = g_list_first (args); it != NULL; it = g_list_next (it)) {
        const gchar *type = (const gchar *) it->data;

        g_string_append_printf (str, "*((global %s *) &cb->data[%zu]) = param%i;\n", type, current, i);
        current += sizes[i];
        i++;
    }

    return g_string_free (str, FALSE);
}

static void
register_callback (App *app, GCallback func, const gchar *name, guint n_params, ...)
{
    #define TYPE_CASE(upper, lower) case G_TYPE_##upper:\
        callback->type_sizes[i] = sizeof (lower);       \
        args = g_list_append (args, G_STRINGIFY (lower));   \
        break;

    Callback *callback;
    gchar *param_list;
    gchar *assignments;
    GList *args = NULL;
    va_list ap;

    callback = g_malloc0 (sizeof (Callback));
    callback->closure = g_cclosure_new (func, NULL, NULL);
    callback->n_params = n_params;
    callback->types = g_malloc0 (sizeof (GType) * n_params);
    callback->type_sizes = g_malloc0 (sizeof (gsize) * n_params);
    callback->size = 0;

    g_closure_set_marshal (callback->closure, g_cclosure_marshal_generic);

    va_start (ap, n_params);

    for (guint i = 0; i < n_params; i++) {
        GType type = va_arg (ap, GType);

        switch (type) {
            TYPE_CASE (FLOAT, float);
            TYPE_CASE (DOUBLE, double);
            TYPE_CASE (INT, int);
            default:
                g_error ("Type not allowed");
        }

        callback->types[i] = type;
        callback->size += callback->type_sizes[i];
    }

    va_end (ap);

    param_list = get_param_list (args);
    assignments = get_assignments (args, callback->type_sizes);

    g_string_append_printf (app->aux_source, "void %s (global Callback *cb, %s)\n", name, param_list);
    g_string_append_printf (app->aux_source, "{\nint idx = get_global_id (0);\nif (idx == 0) {\n%s;\ncb->flag = %i;}}", assignments, app->last);

    g_free (param_list);
    g_free (assignments);

    g_hash_table_insert (app->callbacks, GINT_TO_POINTER (app->last), callback);
    app->last++;

    g_list_free (args);

    #undef TYPE_CASE
}

static gsize
largest_callback_param_size (App *app)
{
    GList *callbacks;
    gsize max_size = 0;

    callbacks = g_hash_table_get_values (app->callbacks);

    for (GList *it = g_list_first (callbacks); it != NULL; it = g_list_next (it)) {
        gsize size = ((Callback *) it->data)->size;

        if (size > max_size)
            max_size = size;
    }

    g_list_free (callbacks);
    return max_size;
}

static void
finish_callback_registration (App *app)
{
    gchar *def;

    def = g_strdup_printf ("typedef struct { unsigned flag; char data[%zu]; } Callback;\n", largest_callback_param_size (app));
    g_string_prepend (app->aux_source, def);
    g_free (def);
}

static void
handle_callback (App *app, guint id, gchar *param_data)
{
    #define TYPE_CASE(upper, lower) case G_TYPE_##upper: \
        g_value_set_##lower (&values[i], *((g##lower *) param_data)); break;

    Callback *callback;
    GValue *values;

    callback = g_hash_table_lookup (app->callbacks, GINT_TO_POINTER (id));
    values = g_malloc0 (callback->n_params * sizeof (GValue));

    for (guint i = 0; i < callback->n_params; i++) {
        GType type = callback->types[i];

        g_value_init (&values[i], type);

        switch (type) {
            TYPE_CASE (FLOAT, float);
            TYPE_CASE (DOUBLE, double);
            TYPE_CASE (INT, int);
            default:
                g_error ("Type not allowed");
        }

        param_data += callback->type_sizes[i];
    }

    /* Calling the closure with fetched params */
    g_closure_invoke (callback->closure, NULL, callback->n_params, values, NULL);

    g_free (values);

    #undef TYPE_CASE
}

static void
listen (App *app)
{
    while (!app->stop) {
        cl_uint *uint_buffer;
        cl_uint callback_id;

        OCL_CHECK_ERROR (clEnqueueReadBuffer (app->queue, app->param_dev_data, CL_TRUE, 0,
                                              app->param_size, app->param_host_data,
                                              0, NULL, NULL));

        uint_buffer = (cl_uint *) app->param_host_data;
        callback_id = uint_buffer[0];

        if (callback_id != 0) {
            handle_callback (app, callback_id, (gchar *) &uint_buffer[1]);

            /* Reset flag */
            uint_buffer[0] = 0;
            OCL_CHECK_ERROR (clEnqueueWriteBuffer (app->queue, app->param_dev_data, CL_TRUE, 0,
                                                   app->param_size, app->param_host_data,
                                                   0, NULL, NULL));

        }
        g_usleep (1000);
    }
}


static void
start_listening (App *app)
{
    cl_int errcode;

    app->param_size = largest_callback_param_size (app);
    app->param_size += sizeof (cl_uint);

    app->param_host_data = g_malloc0 (app->param_size);
    app->param_dev_data = clCreateBuffer (app->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          app->param_size, (gpointer) app->param_host_data, &errcode);
    OCL_CHECK_ERROR (errcode);

#if !(GLIB_CHECK_VERSION (2, 32, 0))
    app->listener = g_thread_create ((GThreadFunc) listen, app, TRUE, NULL);
#else
    app->listener = g_thread_new ("listen", (GThreadFunc) listen, app);
#endif
}

static void
stop_listening (App *app)
{
    g_usleep (2500);
    app->stop = TRUE;
    g_thread_join (app->listener);
}

static cl_program
create_callback_program_from_file (App *app, OclPlatform *ocl,
                                   const gchar *filename, const gchar *options,
                                   cl_int *errcode)
{
    gchar *source;
    gchar *full_source;
    cl_program program;

    source = ocl_read_program (filename);
    full_source = g_strdup_printf ("%s\n%s", app->aux_source->str, source);
    program = ocl_create_program_from_source (ocl, full_source, options, errcode);

    g_free (full_source);
    g_free (source);

    return program;
}

int
main (void)
{
    OclPlatform *ocl;
    cl_int errcode;
    cl_program program;
    App app;
    cl_event event;
    size_t global_work_size = { 1000 };

    ocl = ocl_new_with_queues (0, CL_DEVICE_TYPE_GPU, 0);

    app.stop = FALSE;
    app.aux_source = g_string_new (NULL);
    app.context = ocl_get_context (ocl);
    app.queue = ocl_get_cmd_queues (ocl)[0];

#if !GLIB_CHECK_VERSION(2, 36, 0)
    g_type_init ();
#endif

    app.last = 1;
    app.callbacks = g_hash_table_new (g_direct_hash, g_direct_equal);

    register_callback (&app, (GCallback) handle_print, "print", 3, G_TYPE_FLOAT, G_TYPE_INT, G_TYPE_INT);
    finish_callback_registration (&app);

    program = create_callback_program_from_file (&app, ocl, "callback.cl", NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    app.kernel = clCreateKernel (program, "do_something", &errcode);
    OCL_CHECK_ERROR (errcode);

    start_listening (&app);

    OCL_CHECK_ERROR (clSetKernelArg (app.kernel, 0, sizeof (cl_mem), &app.param_dev_data));
    OCL_CHECK_ERROR (clEnqueueNDRangeKernel (app.queue, app.kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &event));

    clWaitForEvents (1, &event);
    clReleaseEvent (event);

    stop_listening (&app);

    g_string_free (app.aux_source, TRUE);
    clReleaseKernel (app.kernel);
    clReleaseProgram (program);
    ocl_free (ocl);
}
