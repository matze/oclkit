kernel void
fill_ones (global float *out)
{
    out[get_global_id (0)] = 1.0;
}

kernel void
noop (global float *in,
      global float *out)
{
}

kernel void
run_sin (global float *in,
         global float *out)
{
    int idx = get_global_id (0);
    out[idx] = sin(sin((in[idx])));
}
