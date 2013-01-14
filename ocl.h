/*
 *  This file is part of oclkit.
 *
 *  oclkit is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  oclkit is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
 */

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
