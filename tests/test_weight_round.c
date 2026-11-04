/* Unit tests for the weight-precision rounding kernels used by the
 * weight-quantization screening (H3_*_WEIGHT_MODE). Verifies the GPU kernels
 * match the host-side reference implementations bit for bit. */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_weight_round.c: %s\n", message);
    exit(1);
}

static float round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t upper = bits & 0xFFFF0000u;
    uint32_t lower = bits & 0x0000FFFFu;
    if (lower > 0x8000u || (lower == 0x8000u && (upper & 0x00010000u)))
        upper += 0x00010000u;
    float result;
    memcpy(&result, &upper, sizeof(result));
    return result;
}

static void int8_row_reference(float *values, uint64_t rows,
                               uint64_t columns) {
    for (uint64_t row = 0; row < rows; row++) {
        float *line = values + row * columns;
        float amax = 0.0f;
        for (uint64_t column = 0; column < columns; column++) {
            float magnitude = fabsf(line[column]);
            if (magnitude > amax) amax = magnitude;
        }
        float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        float inverse = 1.0f / scale;
        for (uint64_t column = 0; column < columns; column++) {
            float rounded = roundf(line[column] * inverse);
            if (rounded > 127.0f) rounded = 127.0f;
            if (rounded < -127.0f) rounded = -127.0f;
            line[column] = rounded * scale;
        }
    }
}

static uint32_t next_random(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);

    /* BF16 rounding: 300k values spanning many magnitudes. */
    enum { BF16_COUNT = 300000 };
    float *values = malloc(BF16_COUNT * sizeof(*values));
    float *expected = malloc(BF16_COUNT * sizeof(*expected));
    if (!values || !expected) die("out of memory");
    uint32_t state = 7u;
    for (int index = 0; index < BF16_COUNT; index++) {
        uint32_t bits = next_random(&state);
        float value;
        memcpy(&value, &bits, sizeof(value));
        if (isnan(value) || isinf(value)) value = 0.5f;
        values[index] = value;
        expected[index] = round_bf16(value);
    }
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_f32(gpu, values, BF16_COUNT);
    if (!tensor) die("cannot allocate bf16 test tensor");
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_weight_bf16_round_f32(gpu, tensor, BF16_COUNT) ||
        !h3_gpu_submit(gpu)) die("bf16 rounding failed");
    float *got = malloc(BF16_COUNT * sizeof(*got));
    if (!got || !h3_gpu_tensor_read_f32(tensor, got, BF16_COUNT))
        die("cannot read bf16 result");
    uint64_t mismatches = 0;
    for (int index = 0; index < BF16_COUNT; index++)
        if (memcmp(&got[index], &expected[index], sizeof(float)) != 0)
            mismatches++;
    printf("weight bf16 round: %d values, %llu bit mismatches\n",
           BF16_COUNT, (unsigned long long)mismatches);
    if (mismatches) die("bf16 rounding kernel disagrees with host");
    h3_gpu_tensor_free(tensor);
    free(got);

    /* int8 per-row rounding: 128 rows x 777 columns, mixed scales plus an
     * all-zero row and a constant row. */
    enum { ROWS = 128, COLUMNS = 777 };
    values = realloc(values, (size_t)ROWS * COLUMNS * sizeof(*values));
    expected = realloc(expected, (size_t)ROWS * COLUMNS * sizeof(*expected));
    if (!values || !expected) die("out of memory");
    for (int row = 0; row < ROWS; row++) {
        for (int column = 0; column < COLUMNS; column++) {
            size_t index = (size_t)row * COLUMNS + column;
            if (row == 0) {
                values[index] = 0.0f; /* all-zero row */
            } else if (row == 1) {
                values[index] = -0.25f; /* constant row */
            } else {
                float magnitude = powf(10.0f, -6.0f + 12.0f *
                                       ((float)(row % 7)) / 6.0f);
                values[index] = ((float)(next_random(&state) >> 8) /
                                 8388608.0f - 1.0f) * magnitude;
            }
        }
    }
    memcpy(expected, values, (size_t)ROWS * COLUMNS * sizeof(*values));
    int8_row_reference(expected, ROWS, COLUMNS);
    tensor = h3_gpu_tensor_from_f32(gpu, values,
                                    (size_t)ROWS * COLUMNS);
    if (!tensor) die("cannot allocate int8 test tensor");
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_weight_int8_round_f32(gpu, tensor, ROWS, COLUMNS) ||
        !h3_gpu_submit(gpu)) die("int8 rounding failed");
    got = malloc((size_t)ROWS * COLUMNS * sizeof(*got));
    if (!got || !h3_gpu_tensor_read_f32(tensor, got,
                                        (size_t)ROWS * COLUMNS))
        die("cannot read int8 result");
    double worst = 0.0;
    mismatches = 0;
    for (size_t index = 0; index < (size_t)ROWS * COLUMNS; index++) {
        /* Numeric comparison: the GPU kernel can return +0.0 where the host
         * roundf returns -0.0 for tiny negatives, which compares equal. */
        if (got[index] != expected[index]) {
            mismatches++;
            if (mismatches < 5)
                printf("  int8 mismatch at %zu: %.9g vs %.9g\n", index,
                       got[index], expected[index]);
        }
        double delta = fabs((double)got[index] - (double)expected[index]);
        if (delta > worst) worst = delta;
    }
    printf("weight int8 round: %d values, %llu bit mismatches, worst abs %g\n",
           ROWS * COLUMNS, (unsigned long long)mismatches, worst);
    if (mismatches) die("int8 rounding kernel disagrees with host");
    h3_gpu_tensor_free(tensor);
    free(got);
    free(values);
    free(expected);
    h3_gpu_free(gpu);
    printf("PASS tests/test_weight_round.c\n");
    return 0;
}
