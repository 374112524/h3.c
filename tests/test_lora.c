#include "h3_gpu.h"
#include "h3_lora.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HEADS = 2,
    HEAD_DIM = 4,
    INNER = HEADS * HEAD_DIM,
    FFN = 6,
    ROWS = INNER * 3,
    COLUMNS = 16,
    RANK = 2
};

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_lora.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static void write_u64(FILE *file, uint64_t value) {
    unsigned char bytes[8];
    for (unsigned index = 0; index < 8; index++)
        bytes[index] = (unsigned char)((value >> (8 * index)) & 0xffu);
    if (fwrite(bytes, 1, sizeof(bytes), file) != sizeof(bytes))
        fail("cannot write the safetensors prefix");
}

/* Write a minimal safetensors file with one LoRA pair. The down matrix has
 * rank rows and the up matrix has the given number of output rows. */
static void write_lora(const char *path, const char *down, const char *up,
                        float alpha, uint32_t up_rows, const float *a,
                        const float *b) {
    char header[1024];
    int length = snprintf(header, sizeof(header),
        "{\"__metadata__\":{\"alpha\":\"%.4g\",\"steps\":\"4\","
        "\"shift\":\"7\"},"
        "\"%s\":{\"dtype\":\"F32\",\"shape\":[%d,%d],"
        "\"data_offsets\":[0,%d]},"
        "\"%s\":{\"dtype\":\"F32\",\"shape\":[%d,%d],"
        "\"data_offsets\":[%d,%d]}}",
        alpha, down, RANK, COLUMNS, RANK * COLUMNS * 4,
        up, up_rows, RANK, RANK * COLUMNS * 4,
        RANK * COLUMNS * 4 + up_rows * RANK * 4);
    require(length > 0 && (size_t)length < sizeof(header),
            "the safetensors header does not fit");
    FILE *file = fopen(path, "wb");
    if (!file) fail("cannot create the LoRA fixture");
    write_u64(file, (uint64_t)length);
    if (fwrite(header, 1, (size_t)length, file) != (size_t)length)
        fail("cannot write the LoRA header");
    if (fwrite(a, 4, (size_t)RANK * COLUMNS, file) != (size_t)RANK * COLUMNS)
        fail("cannot write the LoRA down matrix");
    if (fwrite(b, 4, (size_t)up_rows * RANK, file) != (size_t)up_rows * RANK)
        fail("cannot write the LoRA up matrix");
    if (fclose(file) != 0) fail("cannot close the LoRA fixture");
}

typedef struct {
    const char *name;
    uint32_t rows;
    uint32_t columns;
    const float *data;
} lora_fixture_tensor;

/* Write a safetensors file with many F32 tensors and alpha metadata. */
static void write_lora_multi(const char *path,
                             const lora_fixture_tensor *tensors,
                             size_t count, float alpha) {
    size_t header_size = 256;
    for (size_t index = 0; index < count; index++)
        header_size += strlen(tensors[index].name) + 96;
    char *header = malloc(header_size);
    if (!header) fail("out of memory building the LoRA fixture header");
    int length = snprintf(header, header_size,
                          "{\"__metadata__\":{\"alpha\":\"%.4g\"},", alpha);
    require(length > 0, "cannot format the LoRA fixture header");
    size_t offset = 0;
    for (size_t index = 0; index < count; index++) {
        size_t bytes = (size_t)tensors[index].rows * tensors[index].columns *
                       sizeof(float);
        int written = snprintf(header + length, header_size - (size_t)length,
                               "%s\"%s\":{\"dtype\":\"F32\",\"shape\":[%u,%u],"
                               "\"data_offsets\":[%zu,%zu]}",
                               index ? "," : "",
                               tensors[index].name, tensors[index].rows,
                               tensors[index].columns, offset, offset + bytes);
        require(written > 0 && (size_t)written < header_size - (size_t)length,
                "the safetensors header does not fit");
        length += written;
        offset += bytes;
    }
    {
        int written = snprintf(header + length, header_size - (size_t)length,
                               "}");
        require(written > 0 && (size_t)written < header_size - (size_t)length,
                "the safetensors header does not fit");
        length += written;
    }
    FILE *file = fopen(path, "wb");
    if (!file) fail("cannot create the LoRA fixture");
    write_u64(file, (uint64_t)length);
    if (fwrite(header, 1, (size_t)length, file) != (size_t)length)
        fail("cannot write the LoRA header");
    for (size_t index = 0; index < count; index++) {
        size_t elements = (size_t)tensors[index].rows * tensors[index].columns;
        if (fwrite(tensors[index].data, sizeof(float), elements, file) !=
            elements)
            fail("cannot write the LoRA fixture data");
    }
    if (fclose(file) != 0) fail("cannot close the LoRA fixture");
    free(header);
}

static float to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x8000u + ((bits >> 16) & 1u);
    uint32_t result = bits & 0xffff0000u;
    float rounded;
    memcpy(&rounded, &result, sizeof(rounded));
    return rounded;
}

static float from_bf16(uint16_t bits) {
    uint32_t value = (uint32_t)bits << 16;
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static h3_gpu_tensor *make_matrix(h3_gpu *gpu, const float *values) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, ROWS * COLUMNS);
    if (!tensor) fail("cannot allocate the LoRA target matrix");
    uint16_t *bits = malloc(sizeof(uint16_t) * ROWS * COLUMNS);
    if (!bits) fail("out of memory preparing the LoRA target matrix");
    for (size_t index = 0; index < ROWS * COLUMNS; index++) {
        uint32_t value;
        float source = to_bf16(values[index]);
        memcpy(&value, &source, sizeof(value));
        bits[index] = (uint16_t)(value >> 16);
    }
    if (!h3_gpu_tensor_write_bf16(tensor, bits, ROWS * COLUMNS))
        fail("cannot fill the LoRA target matrix");
    free(bits);
    return tensor;
}

static float *read_matrix(h3_gpu_tensor *tensor) {
    uint16_t *bits = malloc(sizeof(uint16_t) * ROWS * COLUMNS);
    if (!bits) fail("out of memory reading the merged matrix");
    if (!h3_gpu_tensor_read_bf16(tensor, bits, ROWS * COLUMNS))
        fail("cannot read the merged matrix");
    float *values = malloc(sizeof(float) * ROWS * COLUMNS);
    if (!values) fail("out of memory reading the merged matrix");
    for (size_t index = 0; index < ROWS * COLUMNS; index++)
        values[index] = from_bf16(bits[index]);
    free(bits);
    return values;
}

static void check_close(size_t index, float actual, float expected) {
    float tolerance = 0.02f * fabsf(expected) + 0.002f;
    if (fabsf(actual - expected) > tolerance) {
        fprintf(stderr,
                "FAIL tests/test_lora.c: merged element %zu is %.6f, "
                "expected %.6f\n", index, actual, expected);
        exit(1);
    }
}

/* A fused attention adapter updates every row of the QKV matrix. */
static void test_fused_adapter(h3_gpu *gpu) {
    const char *path = "/tmp/h3_test_lora.safetensors";
    float a[RANK * COLUMNS];
    float b[ROWS * RANK];
    float weight[ROWS * COLUMNS];
    for (size_t index = 0; index < RANK * COLUMNS; index++)
        a[index] = (float)((int)(index % 7) - 3) * 0.25f;
    for (size_t index = 0; index < ROWS * RANK; index++)
        b[index] = (float)((int)(index % 5) - 2) * 0.5f;
    for (size_t index = 0; index < ROWS * COLUMNS; index++)
        weight[index] = (float)((int)(index % 11) - 5) * 0.125f;

    write_lora(path, "blocks.0.attn.qkv_proj.lora_A.weight",
               "blocks.0.attn.qkv_proj.lora_B.weight", 4.0f, ROWS, a, b);

    char error[512];
    h3_lora *lora = h3_lora_load(path, 1.0f, error, sizeof(error));
    if (!lora) fail(error);
    require(h3_lora_active(lora), "the LoRA checkpoint has no adapters");
    require(h3_lora_adapter_count(lora) == 1,
            "the LoRA checkpoint has an unexpected adapter count");
    require(fabsf(h3_lora_alpha(lora) - 4.0f) < 1e-6f,
            "the LoRA alpha was not read from the metadata");
    require(h3_lora_recommended_steps(lora) == 4,
            "the LoRA step recommendation was not read from the metadata");
    require(fabsf(h3_lora_recommended_shift(lora) - 7.0f) < 1e-6f,
            "the LoRA shift recommendation was not read from the metadata");

    h3_gpu_tensor *matrix = make_matrix(gpu, weight);
    h3_lora_layout layout = {HEADS, HEAD_DIM, INNER, FFN};
    if (!h3_gpu_begin(gpu) ||
        !h3_lora_merge(lora, gpu, matrix, "blocks.0.attn",
                       H3_LORA_MODULE_QKV, ROWS, COLUMNS, &layout, error,
                       sizeof(error)) ||
        !h3_gpu_submit(gpu)) fail(error);
    float *merged = read_matrix(matrix);
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t column = 0; column < COLUMNS; column++) {
            float sum = 0.0f;
            for (uint32_t index = 0; index < RANK; index++)
                sum += b[row * RANK + index] * a[index * COLUMNS + column];
            float expected = weight[row * COLUMNS + column] +
                             4.0f / (float)RANK * sum;
            check_close(row * COLUMNS + column,
                        merged[row * COLUMNS + column], expected);
        }
    }
    free(merged);
    h3_gpu_tensor_free(matrix);
    h3_lora_free(lora);
    remove(path);
}

/* A split attention adapter updates only the query rows of the fused QKV
 * matrix, which stores [head][q|k|v][dimension] rows. */
static void test_split_adapter(h3_gpu *gpu) {
    const char *path = "/tmp/h3_test_lora_split.safetensors";
    float a[RANK * COLUMNS];
    float b[INNER * RANK];
    float weight[ROWS * COLUMNS];
    for (size_t index = 0; index < RANK * COLUMNS; index++)
        a[index] = (float)((int)(index % 5) - 2) * 0.5f;
    for (size_t index = 0; index < INNER * RANK; index++)
        b[index] = (float)((int)(index % 3) - 1) * 0.75f;
    for (size_t index = 0; index < ROWS * COLUMNS; index++)
        weight[index] = (float)((int)(index % 9) - 4) * 0.0625f;

    write_lora(path, "transformer.blocks.0.attn.to_q.lora_down.weight",
               "transformer.blocks.0.attn.to_q.lora_up.weight", 2.0f, INNER,
               a, b);

    char error[512];
    h3_lora *lora = h3_lora_load(path, 1.0f, error, sizeof(error));
    if (!lora) fail(error);
    require(h3_lora_adapter_count(lora) == 1,
            "the split LoRA checkpoint has an unexpected adapter count");

    h3_gpu_tensor *matrix = make_matrix(gpu, weight);
    h3_lora_layout layout = {HEADS, HEAD_DIM, INNER, FFN};
    if (!h3_gpu_begin(gpu) ||
        !h3_lora_merge(lora, gpu, matrix, "blocks.0.attn",
                       H3_LORA_MODULE_QKV, ROWS, COLUMNS, &layout, error,
                       sizeof(error)) ||
        !h3_gpu_submit(gpu)) fail(error);
    float *merged = read_matrix(matrix);
    for (uint32_t row = 0; row < ROWS; row++) {
        int is_query = (row % (HEAD_DIM * 3)) < HEAD_DIM;
        uint32_t source = is_query ? row / (HEAD_DIM * 3) * HEAD_DIM +
                                     row % HEAD_DIM : 0;
        for (uint32_t column = 0; column < COLUMNS; column++) {
            float expected = weight[row * COLUMNS + column];
            if (is_query) {
                float sum = 0.0f;
                for (uint32_t index = 0; index < RANK; index++)
                    sum += b[source * RANK + index] *
                           a[index * COLUMNS + column];
                expected += 2.0f / (float)RANK * sum;
            }
            check_close(row * COLUMNS + column,
                        merged[row * COLUMNS + column], expected);
        }
    }
    free(merged);
    h3_gpu_tensor_free(matrix);
    h3_lora_free(lora);
    remove(path);
}

/* Both token-refiner blocks are merged inside a single command buffer by
 * refine_text(), so the merge temporaries of eight matrices must stay alive
 * at once. This reproduces that pattern: a LoRA that adapts every projection
 * of both refiner blocks. */
static void test_refiner_capacity(h3_gpu *gpu) {
    const char *path = "/tmp/h3_test_lora_refiner.safetensors";
    enum { BLOCKS = 2, MATRICES = 4, PAIRS = BLOCKS * MATRICES };
    static const char *suffixes[MATRICES] = {
        "attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"
    };
    static const h3_lora_module kinds[MATRICES] = {
        H3_LORA_MODULE_QKV, H3_LORA_MODULE_OUT,
        H3_LORA_MODULE_FC1, H3_LORA_MODULE_FC2
    };
    lora_fixture_tensor tensors[PAIRS * 2];
    static char names[PAIRS * 2][128];
    float a[PAIRS][RANK * COLUMNS];
    float b[PAIRS][ROWS * RANK];
    float weight[PAIRS][ROWS * COLUMNS];
    for (int pair = 0; pair < PAIRS; pair++) {
        for (size_t index = 0; index < RANK * COLUMNS; index++)
            a[pair][index] = (float)((int)((index + pair) % 7) - 3) * 0.25f;
        for (size_t index = 0; index < ROWS * RANK; index++)
            b[pair][index] = (float)((int)((index + pair) % 5) - 2) * 0.5f;
        for (size_t index = 0; index < ROWS * COLUMNS; index++)
            weight[pair][index] = (float)((int)((index + pair) % 11) - 5) *
                                  0.125f;
        int block = pair / MATRICES;
        int matrix = pair % MATRICES;
        snprintf(names[pair * 2], sizeof(names[pair * 2]),
                 "token_refiner.blocks.%d.%s.lora_A.weight", block,
                 suffixes[matrix]);
        snprintf(names[pair * 2 + 1], sizeof(names[pair * 2 + 1]),
                 "token_refiner.blocks.%d.%s.lora_B.weight", block,
                 suffixes[matrix]);
        tensors[pair * 2].name = names[pair * 2];
        tensors[pair * 2].rows = RANK;
        tensors[pair * 2].columns = COLUMNS;
        tensors[pair * 2].data = a[pair];
        tensors[pair * 2 + 1].name = names[pair * 2 + 1];
        tensors[pair * 2 + 1].rows = ROWS;
        tensors[pair * 2 + 1].columns = RANK;
        tensors[pair * 2 + 1].data = b[pair];
    }
    write_lora_multi(path, tensors, PAIRS * 2, 4.0f);

    char error[512];
    h3_lora *lora = h3_lora_load(path, 1.0f, error, sizeof(error));
    if (!lora) fail(error);
    require(h3_lora_adapter_count(lora) == PAIRS,
            "the refiner LoRA checkpoint has an unexpected adapter count");

    h3_gpu_tensor *matrices[PAIRS];
    for (int pair = 0; pair < PAIRS; pair++) {
        matrices[pair] = make_matrix(gpu, weight[pair]);
        if (!matrices[pair]) fail("cannot allocate a refiner target matrix");
    }
    h3_lora_layout layout = {HEADS, HEAD_DIM, INNER, FFN};
    if (!h3_gpu_begin(gpu)) fail("cannot begin the refiner merge chain");
    for (int pair = 0; pair < PAIRS; pair++) {
        int block = pair / MATRICES;
        int matrix = pair % MATRICES;
        char scope[96];
        snprintf(scope, sizeof(scope), "token_refiner.blocks.%d.%s", block,
                 matrix < 2 ? "attn" : "mlp");
        if (!h3_lora_merge(lora, gpu, matrices[pair], scope, kinds[matrix],
                           ROWS, COLUMNS, &layout, error, sizeof(error)))
            fail(error);
    }
    if (!h3_gpu_submit(gpu)) fail("cannot submit the refiner merge chain");
    h3_lora_flush(lora);
    require(h3_lora_applied_count(lora) == PAIRS,
            "not every refiner adapter was merged");
    for (int pair = 0; pair < PAIRS; pair++) {
        float *merged = read_matrix(matrices[pair]);
        for (uint32_t row = 0; row < ROWS; row++) {
            for (uint32_t column = 0; column < COLUMNS; column++) {
                float sum = 0.0f;
                for (uint32_t index = 0; index < RANK; index++)
                    sum += b[pair][row * RANK + index] *
                           a[pair][index * COLUMNS + column];
                float expected = weight[pair][row * COLUMNS + column] +
                                 4.0f / (float)RANK * sum;
                check_close((size_t)pair * ROWS * COLUMNS + row * COLUMNS +
                            column, merged[row * COLUMNS + column], expected);
            }
        }
        free(merged);
        h3_gpu_tensor_free(matrices[pair]);
    }
    h3_lora_free(lora);
    remove(path);
}

/* Kohya checkpoints without alpha metadata scale every adapter by one
 * (alpha defaults to the rank), including when the ranks differ. */
static void test_rank_alpha_fallback(h3_gpu *gpu) {
    const char *path = "/tmp/h3_test_lora_noalpha.safetensors";
    enum { PAIRS = 2 };
    const uint32_t ranks[PAIRS] = {RANK, 3};
    static char names[PAIRS * 2][128];
    static float a[PAIRS][4 * COLUMNS];
    static float b[PAIRS][ROWS * 4];
    float weight[ROWS * COLUMNS];
    lora_fixture_tensor tensors[PAIRS * 2];
    for (int pair = 0; pair < PAIRS; pair++) {
        uint32_t rank = ranks[pair];
        for (size_t index = 0; index < (size_t)rank * COLUMNS; index++)
            a[pair][index] = (float)((int)((index + pair) % 7) - 3) * 0.25f;
        for (size_t index = 0; index < (size_t)ROWS * rank; index++)
            b[pair][index] = (float)((int)((index + pair) % 5) - 2) * 0.5f;
        snprintf(names[pair * 2], sizeof(names[pair * 2]),
                 "blocks.%d.attn.qkv_proj.lora_A.weight", pair);
        snprintf(names[pair * 2 + 1], sizeof(names[pair * 2 + 1]),
                 "blocks.%d.attn.qkv_proj.lora_B.weight", pair);
        tensors[pair * 2].name = names[pair * 2];
        tensors[pair * 2].rows = rank;
        tensors[pair * 2].columns = COLUMNS;
        tensors[pair * 2].data = a[pair];
        tensors[pair * 2 + 1].name = names[pair * 2 + 1];
        tensors[pair * 2 + 1].rows = ROWS;
        tensors[pair * 2 + 1].columns = rank;
        tensors[pair * 2 + 1].data = b[pair];
    }
    write_lora_multi(path, tensors, PAIRS * 2, 0.0f);

    char error[512];
    h3_lora *lora = h3_lora_load(path, 1.0f, error, sizeof(error));
    if (!lora) fail(error);
    require(h3_lora_adapter_count(lora) == PAIRS,
            "the fallback LoRA checkpoint has an unexpected adapter count");
    for (size_t index = 0; index < ROWS * COLUMNS; index++)
        weight[index] = (float)((int)(index % 11) - 5) * 0.125f;
    /* Both adapters merge into one matrix inside a single command buffer;
     * each must contribute at scale one. */
    h3_gpu_tensor *matrix = make_matrix(gpu, weight);
    h3_lora_layout layout = {HEADS, HEAD_DIM, INNER, FFN};
    if (!h3_gpu_begin(gpu)) fail("cannot begin the fallback merge chain");
    for (int pair = 0; pair < PAIRS; pair++) {
        char scope[64];
        snprintf(scope, sizeof(scope), "blocks.%d.attn", pair);
        if (!h3_lora_merge(lora, gpu, matrix, scope, H3_LORA_MODULE_QKV,
                           ROWS, COLUMNS, &layout, error, sizeof(error)))
            fail(error);
    }
    if (!h3_gpu_submit(gpu)) fail("cannot submit the fallback merge chain");
    h3_lora_flush(lora);
    require(h3_lora_applied_count(lora) == PAIRS,
            "not every fallback adapter was merged");
    float *merged = read_matrix(matrix);
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t column = 0; column < COLUMNS; column++) {
            float sum = 0.0f;
            for (int pair = 0; pair < PAIRS; pair++) {
                uint32_t rank = ranks[pair];
                for (uint32_t index = 0; index < rank; index++)
                    sum += b[pair][row * rank + index] *
                           a[pair][index * COLUMNS + column];
            }
            check_close(row * COLUMNS + column,
                        merged[row * COLUMNS + column],
                        weight[row * COLUMNS + column] + sum);
        }
    }
    free(merged);
    h3_gpu_tensor_free(matrix);
    h3_lora_free(lora);
    remove(path);
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail(error);
    test_fused_adapter(gpu);
    test_split_adapter(gpu);
    test_refiner_capacity(gpu);
    test_rank_alpha_fallback(gpu);
    h3_gpu_free(gpu);
    puts("ok tests/test_lora.c");
    return 0;
}
