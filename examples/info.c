#include <stdio.h>
#include <stdlib.h>
#include "ocl.h"

static char *
get_device_string (cl_device_id device,
                   cl_device_info param)
{
    size_t size;
    char *s;

    OCL_CHECK_ERROR (clGetDeviceInfo (device, param, 0, NULL, &size));
    s = malloc (size);
    OCL_CHECK_ERROR (clGetDeviceInfo (device, param, size, s, NULL));
    return s;
}

static void
show_device_string (const char *fmt,
                    cl_device_id device,
                    cl_device_info param)
{
    char *s;
    s = get_device_string (device, param);
    printf (fmt, s);
    free (s);
}

static void
show_platform_string (const char *fmt,
                      OclPlatform *ocl,
                      cl_platform_info param)
{
    char *s;
    s = ocl_get_platform_info (ocl, param);
    printf (fmt, s);
    free (s);
}

static void
show_context_info (cl_context context)
{
    cl_uint n_formats;
    cl_image_format *formats;

    static const char* channel_orders[] = {
        "CL_R", "CL_A",
        "CL_RG", "CL_RA",
        "CL_RGB", "CL_RGBA",
        "CL_BGRA", "CL_ARGB",
        "CL_INTENSITY", "CL_LUMINANCE",
        "CL_Rx", "CL_RGx",
        "CL_RGBx"
    };

    static const char* channel_types[] = {
        "CL_SNORM_INT8", "CL_SNORM_INT16", "CL_UNORM_INT8", "CL_UNORM_INT16",
        "CL_UNORM_SHORT_565", "CL_UNORM_SHORT_555",
        "CL_UNORM_INT_101010",
        "CL_SIGNED_INT8", "CL_SIGNED_INT16", "CL_SIGNED_INT32",
        "CL_UNSIGNED_INT8", "CL_UNSIGNED_INT16", "CL_UNSIGNED_INT32",
        "CL_HALF_FLOAT", "CL_FLOAT"
    };

    OCL_CHECK_ERROR (clGetSupportedImageFormats (context,
                                                 CL_MEM_READ_WRITE,
                                                 CL_MEM_OBJECT_IMAGE2D,
                                                 0, NULL, &n_formats));

    formats = malloc (n_formats * sizeof (cl_image_format));

    OCL_CHECK_ERROR (clGetSupportedImageFormats (context,
                                                 CL_MEM_READ_WRITE,
                                                 CL_MEM_OBJECT_IMAGE2D,
                                                 n_formats, formats, NULL));

    printf ("Supported image formats\n\n");

    for (unsigned i = 0; i < n_formats; i++) {
        cl_image_format *format = &formats[i];

        printf (" %s:%s", channel_orders[format->image_channel_order - CL_R],
                          channel_types[format->image_channel_data_type - CL_SNORM_INT8]);

        if ((i + 1) % 4 == 0)
            printf ("\n");
    }

    printf ("\n\n");

    free (formats);
}

#define show_device_scalar(fmt, device, param, type) {\
    void *mem; \
    mem = malloc (sizeof(type)); \
    clGetDeviceInfo(device, param, sizeof(type), mem, NULL); \
    printf (fmt, *((type *) mem)); \
    free (mem); }

static void
show_device_info (cl_device_id device)
{
    char *name;
    char *profile;
    size_t cube[3];

    name = get_device_string (device, CL_DEVICE_NAME);
    profile = get_device_string (device, CL_DEVICE_PROFILE);
    printf ("OpenCL device `%s': %s\n\n", name, profile);

    show_device_string ("  Vendor:\t%s\n", device, CL_DEVICE_VENDOR);
    show_device_string ("  Version:\t%s\n", device, CL_DEVICE_VERSION);
    show_device_string ("  Extensions:\t%s\n\n", device, CL_DEVICE_EXTENSIONS);

    show_device_scalar ("  MAX_MEM_ALLOC_SIZE:\t%lu bytes\n", device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong);
    show_device_scalar ("  MAX_COMPUTE_UNITS:\t%i units\n", device, CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint);
    show_device_scalar ("  MAX_WORK_GROUP_SIZE:\t%zu work items\n", device, CL_DEVICE_MAX_WORK_GROUP_SIZE, size_t);
    show_device_scalar ("  LOCAL_MEM_SIZE:\t%lu bytes\n", device, CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong);

    OCL_CHECK_ERROR (clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof (cube), &cube, NULL));
    printf ("  MAX_WORK_ITEM_SIZES: (%zu, %zu, %zu)\n", cube[0], cube[1], cube[2]);


    free (name);
    free (profile);
}

int
main (void)
{
    OclPlatform *ocl;
    cl_device_id *devices;
    int n_devices;

    ocl = ocl_new (CL_DEVICE_TYPE_ALL, 1);

    if (ocl == NULL)
        return 1;

    show_platform_string ("Platform name: %s\n", ocl, CL_PLATFORM_NAME);
    show_platform_string ("Platform vendor: %s\n\n", ocl, CL_PLATFORM_VENDOR);

    show_context_info (ocl_get_context (ocl));

    n_devices = ocl_get_num_devices (ocl);
    devices = ocl_get_devices (ocl);

    for (int i = 0; i < n_devices; i++) {
        show_device_info (devices[i]);

        if (i < n_devices - 1)
            printf ("\n\n");
    }

    ocl_free (ocl);

    return 0;
}
