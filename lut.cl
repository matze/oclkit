__kernel void
fill_with_lut (__constant float *sin_lut,
               __constant float *cos_lut,
               __global float *output)
{
    int idx = get_global_id (0);
    int idy = get_global_id (1);
    int width = get_global_size (0);

    float sum = 0.0f;

    for (int i = 0; i < idx; i++) {
        sum += cos_lut[i] * sin_lut[idy];
    }

    output[idy * width + idx] = sum;
}

__kernel void
fill_with_comp (__global float *output)
{
    int idx = get_global_id (0);
    int idy = get_global_id (1);
    int width = get_global_size (0);

    float sum = 0.0f;

    for (int i = 0; i < idx; i++) {
        sum += cos((float) i) * sin((float) idy);
    }

    output[idy * width + idx] = sum;
}

