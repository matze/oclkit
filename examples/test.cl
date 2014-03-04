__kernel void
fill_ones (__global float *out)
{
    out[get_global_id (0)] = 1.0;
}

__kernel void
run_sin (__global float *in,
         __global float *out)
{
    int idx = get_global_id (0);
    out[idx] = sin(sin((in[idx])));
}
