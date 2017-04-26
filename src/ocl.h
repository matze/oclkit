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
 *  along with oclkit.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OCL_H
#define OCL_H

#include <CL/cl.h>
#include <stdio.h>

#include "AOCL_Utils.h"

using namespace aocl_utils;

typedef struct OclPlatform OclPlatform;

#define OCL_CHECK_ERROR(error) { \
    if ((error) != CL_SUCCESS) fprintf (stderr, "OpenCL error <%s:%i>: %s\n", __FILE__, __LINE__, ocl_strerr((error))); }

int                 ocl_read_args       (int                 argc,
                                         const char        **argv,
                                         unsigned int       *platform,
                                         cl_device_type     *type);
OclPlatform *       ocl_new             (unsigned            platform,
                                         cl_device_type      type);
OclPlatform *       ocl_new_with_queues (unsigned            platform,
                                         cl_device_type      type,
                                         cl_command_queue_properties
                                                             queue_properties);
OclPlatform *       ocl_new_from_args   (int                 argc,
                                         const char **       argv,
                                         cl_command_queue_properties
                                                             queue_properties);
OclPlatform *       ocl_new_from_args_bare
                                        (int                 argc,
                                         const char        **argv);
void                ocl_print_usage     (void);
void                ocl_free            (OclPlatform        *ocl);
char *              ocl_get_platform_info
                                        (OclPlatform        *ocl,
                                         cl_platform_info    param);
cl_context          ocl_get_context     (OclPlatform        *ocl);
cl_program          ocl_create_program_from_file
                                        (OclPlatform        *ocl,
                                         const char         *filename,
                                         const char         *options,
                                         cl_int             *errcode);
cl_program          ocl_create_program_from_source
                                        (OclPlatform        *ocl,
                                         const char         *source,
                                         const char         *options,
                                         cl_int             *errcode);
cl_program          ocl_create_program_from_binary
                                        (OclPlatform        *ocl,
                                         const char         *source,
                                         cl_int             *errcode);
cl_program          ocl_create_program_from_binary_for_fpga
                                        (OclPlatform        *ocl,
                                         char* binaryfile);
int                 ocl_get_num_devices (OclPlatform        *ocl);
cl_device_id *      ocl_get_devices     (OclPlatform        *ocl);
cl_command_queue *  ocl_get_cmd_queues  (OclPlatform        *ocl);
const char*         ocl_strerr          (int                 error);
char*               ocl_read_program    (const char         *filename);
void                ocl_get_event_times (cl_event            event,
                                         cl_ulong           *start,
                                         cl_ulong           *end,
                                         cl_ulong           *queued,
                                         cl_ulong           *submitted);


#endif
