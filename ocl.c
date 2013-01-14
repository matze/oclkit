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

#include <stdio.h>
#include <assert.h>
#include "ocl.h"

struct OclPlatform {
    cl_context           context;
    cl_uint              num_devices;
    cl_device_id        *devices;
    cl_command_queue    *cmd_queues;
};

static const char* opencl_error_msgs[] = {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "CL_MISALIGNED_SUB_BUFFER_OFFSET",
    "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",

    /* next IDs start at 30! */
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE"
};

const char*
ocl_strerr (int error)
{
    if (error >= -14)
        return opencl_error_msgs[-error];
    if (error <= -30)
        return opencl_error_msgs[-error-15];

    return NULL;
}

static char *
ocl_read_program (const char *filename)
{
    FILE *fp;
    char *buffer;
    size_t length;
    size_t buffer_length;
    
    if ((fp = fopen(filename, "r")) == NULL)
        return NULL;

    fseek (fp, 0, SEEK_END);
    length = ftell (fp);
    rewind (fp);

    buffer = malloc (length + 1);
    buffer[length] = '\0';

    if (buffer == NULL) {
        fclose(fp);
        return NULL;
    }

    buffer_length = fread (buffer, 1, length, fp);
    fclose(fp);

    if (buffer_length != length) {
        free (buffer);
        buffer = NULL;
    }

    return buffer;
}

OclPlatform *
ocl_new (void)
{
    OclPlatform *ocl;
    cl_platform_id platform;
    int errcode;
    
    OCL_CHECK_ERROR (clGetPlatformIDs (1, &platform, NULL));

    if (platform == NULL)
        return NULL;

    ocl = malloc (sizeof(OclPlatform));

    OCL_CHECK_ERROR (clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 0, NULL, &ocl->num_devices));
    ocl->devices = malloc (ocl->num_devices * sizeof(cl_device_id));
    OCL_CHECK_ERROR (clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, ocl->num_devices, ocl->devices, NULL));

    ocl->context = clCreateContext (NULL, ocl->num_devices, ocl->devices, NULL, NULL, &errcode);
    OCL_CHECK_ERROR(errcode);

    ocl->cmd_queues = malloc (ocl->num_devices * sizeof(cl_command_queue));

    for (int i = 0; i < ocl->num_devices; i++) {
        ocl->cmd_queues[i] = clCreateCommandQueue (ocl->context, ocl->devices[i], 0, &errcode);
        OCL_CHECK_ERROR (errcode);
    }

    return ocl;
}

void
ocl_free (OclPlatform *ocl)
{
    for (int i = 0; i < ocl->num_devices; i++)
        OCL_CHECK_ERROR (clReleaseCommandQueue (ocl->cmd_queues[i]));

    OCL_CHECK_ERROR (clReleaseContext (ocl->context));

    free (ocl->devices);
    free (ocl->cmd_queues);
    free (ocl);
}

cl_program
ocl_get_program (OclPlatform *ocl,
                 const char *filename,
                 const char *options)
{
    char *buffer;
    int errcode;
    cl_program program;
    
    buffer = ocl_read_program(filename);

    if (buffer == NULL)
        return NULL;

    program = clCreateProgramWithSource (ocl->context, 1, (const char **) &buffer, NULL, &errcode);
    OCL_CHECK_ERROR (errcode);

    if (errcode != CL_SUCCESS) {
        free(buffer);
        return NULL;
    }

    errcode = clBuildProgram (program, ocl->num_devices, ocl->devices, options, NULL, NULL);
    OCL_CHECK_ERROR (errcode);

    if (errcode != CL_SUCCESS) {
        const int LOG_SIZE = 4096;
        char* log;
        
        log = malloc (LOG_SIZE * sizeof(char));
        OCL_CHECK_ERROR (clGetProgramBuildInfo (program, ocl->devices[0],
                                                CL_PROGRAM_BUILD_LOG, LOG_SIZE,
                                                (void*) log, NULL));

        fprintf (stderr, "\n=== Build log for %s===%s\n\n", filename, log);
        free (log);
        free (buffer);
        return NULL;
    }

    free(buffer);
    return program;
}

cl_context
ocl_get_context (OclPlatform *ocl)
{
    assert (ocl != NULL);
    return ocl->context;
}

int
ocl_get_num_devices (OclPlatform *ocl)
{
    assert (ocl != NULL);
    return ocl->num_devices;
}

cl_device_id *
ocl_get_devices (OclPlatform *ocl)
{
    assert (ocl != NULL);
    return ocl->devices;
}

cl_command_queue *
ocl_get_cmd_queues (OclPlatform *ocl)
{
    assert (ocl != NULL);
    return ocl->cmd_queues;
}
