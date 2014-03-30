__kernel void assign(__global float *input, __global float *output)
{
    const int idx = get_global_id (0); 
    input[idx] = output[idx]; 
}

__kernel void two_const_params(__constant float *c_param_1,
                               __constant float *c_param_2)
{
}

__kernel void three_const_params(__constant float *c_param_1,
                                 __constant float *c_param_2,
                                 __constant float *c_param_3)
{
}

__kernel void four_const_params(__constant float *c_param_1,
                                __constant float *c_param_2,
                                __constant float *c_param_3,
                                __constant float *c_param_4)
{
}

__kernel void two_local_params(__constant float *l_param_1,
                               __constant float *l_param_2)
{
}

__kernel void three_local_params(__constant float *l_param_1,
                                 __constant float *l_param_2,
                                 __constant float *l_param_3)
{
}

__kernel void four_local_params(__constant float *l_param_1,
                                __constant float *l_param_2,
                                __constant float *l_param_3,
                                __constant float *l_param_4)
{
}

__kernel void two_global_params(__global float *g_param_1,
                                __global float *g_param_2)
{
}

__kernel void three_global_params(__global float *g_param_1,
                                  __global float *g_param_2,
                                  __global float *g_param_3)
{
}

__kernel void four_global_params(__global float *g_param_1,
                                 __global float *g_param_2,
                                 __global float *g_param_3,
                                 __global float *g_param_4)
{
}

__kernel void global_local_constant_params(__global   float *g_param_1,
                                           __local    float *l_param2,
                                           __constant float *c_param3,
                                           __constant float *c_param4)
{
}
