__kernel void
fill_ones (__global float *out)
{
    out[get_global_id (0)] = 1.0;
}
