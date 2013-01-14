#ifndef OCL_H
#define OCL_H

#include <CL/cl.h>

typedef struct OclPlatform OclPlatform;

#define OCL_CHECK_ERROR(error) { \
    if ((error) != CL_SUCCESS) fprintf (stderr, "OpenCL error <%s:%i>: %s", __FILE__, __LINE__, ocl_strerr((error))); }

OclPlatform *       ocl_new             (void);
void                ocl_free            (OclPlatform    *ocl);
cl_context          ocl_get_context     (OclPlatform    *ocl);
cl_program          ocl_get_program     (OclPlatform    *ocl,
                                         const char     *filename,
                                         const char     *options);
int                 ocl_get_num_devices (OclPlatform    *ocl);
cl_device_id *      ocl_get_devices     (OclPlatform    *ocl);
cl_command_queue *  ocl_get_cmd_queues  (OclPlatform    *ocl);
const char*         ocl_strerr          (int error);


#endif
