#include "h3_lora.h"

#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define H3_LORA_MAX_TOKENS 32
#define H3_LORA_PATH_MAX 192
#define H3_LORA_NAME_MAX 320
#define H3_LORA_PENDING_INITIAL 8

typedef enum {
    H3_LORA_SLOT_FULL = 0,
    H3_LORA_SLOT_Q,
    H3_LORA_SLOT_K,
    H3_LORA_SLOT_V,
    H3_LORA_SLOT_GATE,
    H3_LORA_SLOT_UP
} h3_lora_slot;

typedef struct {
    /* Scope of the adapted matrix, for example "blocks.3.attn". Adapters
     * that name a split projection (q, k, v, gate, up) share the scope of
     * the fused matrix they belong to. */
    char scope[H3_LORA_PATH_MAX];
    h3_lora_module module_kind;
    h3_lora_slot slot;
    uint32_t rank;
    uint32_t b_rank;
    uint32_t rows;
    uint32_t columns;
    float *a;
    float *b;
    int have_a;
    int have_b;
} h3_lora_adapter;

struct h3_lora {
    char *path;
    float scale;
    float alpha;
    int has_alpha;
    int alpha_from_rank;
    int recommended_steps;
    float recommended_shift;
    h3_lora_adapter *adapters;
    size_t adapter_count;
    size_t usable_count;
    /* Merge temporaries stay alive until the command buffer that uses them
     * has been committed and completed. The list grows on demand: a single
     * command buffer can merge eight matrices (both token refiner blocks),
     * i.e. sixteen temporaries at once. */
    h3_gpu_tensor **pending;
    size_t pending_count;
    size_t pending_capacity;
    size_t merge_count;
    double merge_seconds;
    size_t tensor_count;
    size_t skipped;
    size_t applied;
};

static double h3_lora_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float bf16_to_float(uint16_t bits) {
    uint32_t value = ((uint32_t)bits) << 16;
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static float f16_to_float(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x3ffu;
    uint32_t value;
    if (exponent == 0) {
        if (!mantissa) {
            value = sign;
        } else {
            exponent = 127 - 15 + 1;
            while (!(mantissa & 0x400u)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3ffu;
            value = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | (mantissa << 13);
    } else {
        value = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static int token_is(const char *token, const char *value) {
    return !strcmp(token, value);
}

static size_t split_tokens(char *text, char *tokens[H3_LORA_MAX_TOKENS]) {
    size_t count = 0;
    char *cursor = text;
    while (cursor && count < H3_LORA_MAX_TOKENS) {
        char *dot = strchr(cursor, '.');
        if (dot) *dot = '\0';
        tokens[count++] = cursor;
        cursor = dot ? dot + 1 : NULL;
    }
    return count;
}

/* Split a LoRA tensor name into its module path and its A/B role. Returns 1
 * when the name carries a low-rank marker, 0 when the tensor belongs to
 * something else, and -1 for malformed names. */
static int lora_split_name(const char *name, char *module, size_t module_size,
                            int *is_a) {
    char copy[H3_LORA_NAME_MAX];
    if (!name || strlen(name) >= sizeof(copy)) return 0;
    snprintf(copy, sizeof(copy), "%s", name);
    char *tokens[H3_LORA_MAX_TOKENS];
    size_t count = split_tokens(copy, tokens);
    if (!count) return 0;
    int marker = -1;
    int role_a = 0;
    for (size_t index = 0; index < count && marker < 0; index++) {
        const char *token = tokens[index];
        if (token_is(token, "lora_A") || token_is(token, "lora_a") ||
            token_is(token, "lora_down")) {
            marker = (int)index;
            role_a = 1;
        } else if (token_is(token, "lora_B") || token_is(token, "lora_b") ||
                   token_is(token, "lora_up")) {
            marker = (int)index;
            role_a = 0;
        } else if (token_is(token, "lora") && index + 1 < count &&
                   (token_is(tokens[index + 1], "down") ||
                    token_is(tokens[index + 1], "up"))) {
            marker = (int)index + 1;
            role_a = token_is(tokens[index + 1], "down");
        } else if ((token_is(token, "down") || token_is(token, "up")) &&
                   index > 0 && token_is(tokens[index - 1], "lora")) {
            marker = (int)index;
            role_a = token_is(token, "down");
        } else if ((token_is(token, "A") || token_is(token, "B")) &&
                   (index + 1 == count || token_is(tokens[index + 1], "weight") ||
                    token_is(tokens[index + 1], "scale"))) {
            marker = (int)index;
            role_a = token_is(token, "A");
        } else if ((token_is(token, "down") || token_is(token, "up")) &&
                   (index + 1 == count ||
                    token_is(tokens[index + 1], "weight"))) {
            marker = (int)index;
            role_a = token_is(token, "down");
        }
    }
    if (marker <= 0) return 0;
    size_t length = 0;
    module[0] = '\0';
    for (size_t index = 0; index < (size_t)marker; index++) {
        const char *token = tokens[index];
        if (index + 1 == (size_t)marker && token_is(token, "weight")) continue;
        size_t token_length = strlen(token);
        if (length + token_length + 2 > module_size) return -1;
        if (length) module[length++] = '.';
        memcpy(module + length, token, token_length);
        length += token_length;
        module[length] = '\0';
    }
    if (!length) return 0;
    *is_a = role_a;
    return 1;
}

static void strip_prefix(char *module) {
    static const char *prefixes[] = {
        "base_model.model.", "diffusion_model.", "transformer.",
        "base_model.", "lora_unet.", "unet.", "model."
    };
    for (size_t index = 0; index < sizeof(prefixes) / sizeof(prefixes[0]);
         index++) {
        size_t length = strlen(prefixes[index]);
        if (!strncmp(module, prefixes[index], length)) {
            memmove(module, module + length, strlen(module + length) + 1);
            return;
        }
    }
}

/* Map a normalized module path onto the fused H3 projection it adapts. The
 * consumed count is the number of trailing path components that name the
 * projection itself, so the remaining prefix is the adapter scope. */
static int lora_module_kind(const char *module, h3_lora_module *kind,
                             h3_lora_slot *slot, size_t *consumed) {
    char copy[H3_LORA_PATH_MAX];
    snprintf(copy, sizeof(copy), "%s", module);
    char *tokens[H3_LORA_MAX_TOKENS];
    size_t count = split_tokens(copy, tokens);
    if (!count) return 0;
    const char *last = tokens[count - 1];
    const char *previous = count > 1 ? tokens[count - 2] : NULL;
    *consumed = 1;
    if (token_is(last, "qkv_proj") || token_is(last, "qkv") ||
        token_is(last, "to_qkv") || token_is(last, "qkv_linear")) {
        *kind = H3_LORA_MODULE_QKV;
        *slot = H3_LORA_SLOT_FULL;
        return 1;
    }
    if (token_is(last, "to_q") || token_is(last, "q_proj") ||
        token_is(last, "q")) {
        *kind = H3_LORA_MODULE_QKV;
        *slot = H3_LORA_SLOT_Q;
        return 1;
    }
    if (token_is(last, "to_k") || token_is(last, "k_proj") ||
        token_is(last, "k")) {
        *kind = H3_LORA_MODULE_QKV;
        *slot = H3_LORA_SLOT_K;
        return 1;
    }
    if (token_is(last, "to_v") || token_is(last, "v_proj") ||
        token_is(last, "v")) {
        *kind = H3_LORA_MODULE_QKV;
        *slot = H3_LORA_SLOT_V;
        return 1;
    }
    if (token_is(last, "out_proj") || token_is(last, "o_proj") ||
        token_is(last, "to_out")) {
        *kind = H3_LORA_MODULE_OUT;
        *slot = H3_LORA_SLOT_FULL;
        return 1;
    }
    if (token_is(last, "0") && previous && token_is(previous, "to_out")) {
        *kind = H3_LORA_MODULE_OUT;
        *slot = H3_LORA_SLOT_FULL;
        *consumed = 2;
        return 1;
    }
    if (token_is(last, "fc1")) {
        *kind = H3_LORA_MODULE_FC1;
        *slot = H3_LORA_SLOT_FULL;
        return 1;
    }
    if (token_is(last, "fc2") || token_is(last, "down_proj") ||
        token_is(last, "w2")) {
        *kind = H3_LORA_MODULE_FC2;
        *slot = H3_LORA_SLOT_FULL;
        return 1;
    }
    if (token_is(last, "gate_proj") || token_is(last, "w1") ||
        token_is(last, "gate")) {
        *kind = H3_LORA_MODULE_FC1;
        *slot = H3_LORA_SLOT_GATE;
        return 1;
    }
    if (token_is(last, "up_proj") || token_is(last, "w3") ||
        token_is(last, "up")) {
        *kind = H3_LORA_MODULE_FC1;
        *slot = H3_LORA_SLOT_UP;
        return 1;
    }
    return 0;
}

/* Normalize the family component of a scope so that "attention" and "attn"
 * or "feed_forward" and "mlp" describe the same block. */
static void normalize_scope(char *scope) {
    char *last = strrchr(scope, '.');
    if (!last) return;
    char *token = last + 1;
    if (token_is(token, "attention") || token_is(token, "self_attn") ||
        token_is(token, "attn")) {
        snprintf(token, strlen(token) + 1, "attn");
    } else if (token_is(token, "feed_forward") || token_is(token, "ff") ||
               token_is(token, "ffn") || token_is(token, "mlp")) {
        snprintf(token, strlen(token) + 1, "mlp");
    }
}

static h3_lora_adapter *lora_find_adapter(h3_lora *lora, const char *scope,
                                           h3_lora_module kind,
                                           h3_lora_slot slot) {
    for (size_t index = 0; index < lora->adapter_count; index++) {
        h3_lora_adapter *adapter = &lora->adapters[index];
        if (adapter->module_kind == kind && adapter->slot == slot &&
            !strcmp(adapter->scope, scope)) return adapter;
    }
    h3_lora_adapter *grown = realloc(lora->adapters,
                                     (lora->adapter_count + 1) *
                                     sizeof(*grown));
    if (!grown) return NULL;
    lora->adapters = grown;
    h3_lora_adapter *adapter = &lora->adapters[lora->adapter_count++];
    memset(adapter, 0, sizeof(*adapter));
    snprintf(adapter->scope, sizeof(adapter->scope), "%s", scope);
    adapter->module_kind = kind;
    adapter->slot = slot;
    return adapter;
}

static int lora_read_matrix(const h3_st_header *header,
                             const h3_st_tensor *tensor, float **values) {
    uint64_t elements = h3_st_tensor_elements(tensor);
    if (!elements || elements > SIZE_MAX / sizeof(float)) return 0;
    float *data = malloc((size_t)elements * sizeof(float));
    if (!data) return 0;
    if (tensor->dtype == H3_DTYPE_F32) {
        if (!h3_st_read_data(header, tensor, data,
                             (size_t)elements * sizeof(float), NULL, 0)) {
            free(data);
            return 0;
        }
    } else if (tensor->dtype == H3_DTYPE_BF16 || tensor->dtype == H3_DTYPE_F16) {
        uint16_t *raw = malloc((size_t)elements * sizeof(uint16_t));
        if (!raw || !h3_st_read_data(header, tensor, raw,
                                     (size_t)elements * sizeof(uint16_t),
                                     NULL, 0)) {
            free(raw);
            free(data);
            return 0;
        }
        for (uint64_t index = 0; index < elements; index++) {
            data[index] = tensor->dtype == H3_DTYPE_BF16 ?
                bf16_to_float(raw[index]) : f16_to_float(raw[index]);
        }
        free(raw);
    } else {
        free(data);
        return 0;
    }
    *values = data;
    return 1;
}

/* Read the step count that a step-acceleration checkpoint recommends. */
static int lora_metadata_steps(const h3_st_header *header, int *steps) {
    static const char *keys[] = {"steps", "num_steps", "acceleration_steps",
                                 "recommended_steps", "timesteps"};
    for (size_t index = 0; index < sizeof(keys) / sizeof(keys[0]); index++) {
        const char *value = h3_st_metadata(header, keys[index]);
        if (!value || !*value) continue;
        char *end = NULL;
        long parsed = strtol(value, &end, 10);
        if (end == value || *end || parsed < 1 || parsed > 1000) continue;
        *steps = (int)parsed;
        return 1;
    }
    return 0;
}

/* Read the flow shift that a step-acceleration checkpoint was trained with. */
static int lora_metadata_shift(const h3_st_header *header, float *shift) {
    static const char *keys[] = {"shift", "sigma_shift", "flow_shift",
                                 "timestep_shift"};
    for (size_t index = 0; index < sizeof(keys) / sizeof(keys[0]); index++) {
        const char *value = h3_st_metadata(header, keys[index]);
        if (!value || !*value) continue;
        char *end = NULL;
        double parsed = strtod(value, &end);
        if (end == value || *end || !isfinite(parsed) || parsed <= 0.0 ||
            parsed > 100.0) continue;
        *shift = (float)parsed;
        return 1;
    }
    return 0;
}

static int lora_metadata_alpha(const h3_st_header *header, float *alpha) {
    static const char *keys[] = {"alpha", "network_alpha", "lora_alpha"};
    for (size_t index = 0; index < sizeof(keys) / sizeof(keys[0]); index++) {
        const char *value = h3_st_metadata(header, keys[index]);
        if (!value || !*value) continue;
        char *end = NULL;
        double parsed = strtod(value, &end);
        if (end == value || !isfinite(parsed) || parsed <= 0.0) continue;
        *alpha = (float)parsed;
        return 1;
    }
    return 0;
}

h3_lora *h3_lora_load(const char *path, float scale, char *error,
                       size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path) {
        fail(error, error_size, "a LoRA checkpoint path is required");
        return NULL;
    }
    if (!isfinite(scale) || scale < 0.0f) {
        fail(error, error_size, "the LoRA scale must be a positive number");
        return NULL;
    }
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, error_size)) return NULL;
    h3_lora *lora = calloc(1, sizeof(*lora));
    if (!lora) {
        h3_st_free_header(&header);
        fail(error, error_size, "out of memory loading the LoRA checkpoint");
        return NULL;
    }
    lora->path = strdup(path);
    lora->scale = scale;
    if (!lora->path) {
        h3_st_free_header(&header);
        h3_lora_free(lora);
        fail(error, error_size, "out of memory loading the LoRA checkpoint");
        return NULL;
    }
    lora_metadata_steps(&header, &lora->recommended_steps);
    lora_metadata_shift(&header, &lora->recommended_shift);
    float alpha = 0.0f;
    if (lora_metadata_alpha(&header, &alpha)) {
        lora->alpha = alpha;
        lora->has_alpha = 1;
    } else {
        const h3_st_tensor *tensor = h3_st_find(&header, "alpha");
        if (!tensor) tensor = h3_st_find(&header, "network_alpha");
        if (tensor && h3_st_tensor_elements(tensor) == 1) {
            float *value = NULL;
            if (lora_read_matrix(&header, tensor, &value) && value) {
                lora->alpha = value[0];
                lora->has_alpha = lora->alpha > 0.0f;
                free(value);
            }
        }
    }
    for (size_t index = 0; index < header.tensor_count; index++) {
        const h3_st_tensor *tensor = &header.tensors[index];
        lora->tensor_count++;
        char module[H3_LORA_PATH_MAX];
        int is_a = 0;
        int split = lora_split_name(tensor->name, module, sizeof(module),
                                    &is_a);
        if (split <= 0) {
            lora->skipped++;
            continue;
        }
        strip_prefix(module);
        h3_lora_module kind;
        h3_lora_slot slot;
        size_t consumed = 0;
        if (!lora_module_kind(module, &kind, &slot, &consumed)) {
            lora->skipped++;
            continue;
        }
        char scope[H3_LORA_PATH_MAX];
        snprintf(scope, sizeof(scope), "%s", module);
        for (size_t step = 0; step < consumed; step++) {
            char *last = strrchr(scope, '.');
            if (!last) {
                scope[0] = '\0';
                break;
            }
            *last = '\0';
        }
        if (!scope[0]) {
            lora->skipped++;
            continue;
        }
        normalize_scope(scope);
        if (tensor->ndim != 2) {
            lora->skipped++;
            continue;
        }
        uint32_t rows = (uint32_t)tensor->shape[0];
        uint32_t columns = (uint32_t)tensor->shape[1];
        float *values = NULL;
        if (!lora_read_matrix(&header, tensor, &values)) {
            lora->skipped++;
            continue;
        }
        h3_lora_adapter *adapter = lora_find_adapter(lora, scope, kind, slot);
        if (!adapter) {
            free(values);
            lora->skipped++;
            continue;
        }
        if (is_a) {
            /* The down matrix is [rank, input channels]. */
            free(adapter->a);
            adapter->a = values;
            adapter->rank = rows;
            adapter->columns = columns;
            adapter->have_a = 1;
        } else {
            /* The up matrix is [output rows, rank]. */
            free(adapter->b);
            adapter->b = values;
            adapter->rows = rows;
            adapter->b_rank = columns;
            adapter->have_b = 1;
        }
    }
    h3_st_free_header(&header);
    size_t usable = 0;
    uint32_t first_rank = 0;
    for (size_t index = 0; index < lora->adapter_count; index++) {
        h3_lora_adapter *adapter = &lora->adapters[index];
        if (!adapter->have_a || !adapter->have_b || !adapter->a ||
            !adapter->b || !adapter->rank || adapter->b_rank != adapter->rank) {
            free(adapter->a);
            free(adapter->b);
            adapter->a = NULL;
            adapter->b = NULL;
            adapter->have_a = 0;
            adapter->have_b = 0;
            continue;
        }
        if (!first_rank) first_rank = adapter->rank;
        usable++;
    }
    lora->usable_count = usable;
    if (!usable) {
        fail(error, error_size,
             "%s: no usable H3 DiT LoRA adapters were found", path);
        h3_lora_free(lora);
        return NULL;
    }
    if (!lora->has_alpha) {
        /* Kohya checkpoints without metadata use the rank as the alpha, so
         * every adapter scales by one. The reported alpha stays the first
         * adapter's rank for the common uniform-rank checkpoint. */
        lora->alpha = (float)first_rank;
        lora->alpha_from_rank = 1;
        lora->has_alpha = 1;
    }
    return lora;
}

void h3_lora_free(h3_lora *lora) {
    if (!lora) return;
    h3_lora_flush(lora);
    for (size_t index = 0; index < lora->adapter_count; index++) {
        free(lora->adapters[index].a);
        free(lora->adapters[index].b);
    }
    free(lora->adapters);
    free(lora->pending);
    free(lora->path);
    free(lora);
}

int h3_lora_active(const h3_lora *lora) {
    return lora && lora->usable_count > 0;
}

size_t h3_lora_adapter_count(const h3_lora *lora) {
    return lora ? lora->usable_count : 0;
}

size_t h3_lora_tensor_count(const h3_lora *lora) {
    return lora ? lora->tensor_count : 0;
}

size_t h3_lora_skipped_count(const h3_lora *lora) {
    return lora ? lora->skipped : 0;
}

size_t h3_lora_applied_count(const h3_lora *lora) {
    return lora ? lora->applied : 0;
}

size_t h3_lora_merge_count(const h3_lora *lora) {
    return lora ? lora->merge_count : 0;
}

double h3_lora_merge_seconds(const h3_lora *lora) {
    return lora ? lora->merge_seconds : 0.0;
}

int h3_lora_has_scope(const h3_lora *lora, const char *scope) {
    if (!lora || !scope) return 0;
    for (size_t index = 0; index < lora->adapter_count; index++) {
        if (!strcmp(lora->adapters[index].scope, scope)) return 1;
    }
    return 0;
}

const char *h3_lora_path(const h3_lora *lora) {
    return lora ? lora->path : NULL;
}

float h3_lora_scale(const h3_lora *lora) {
    return lora ? lora->scale : 0.0f;
}

float h3_lora_alpha(const h3_lora *lora) {
    return lora ? lora->alpha : 0.0f;
}

int h3_lora_recommended_steps(const h3_lora *lora) {
    return lora ? lora->recommended_steps : 0;
}

float h3_lora_recommended_shift(const h3_lora *lora) {
    return lora ? lora->recommended_shift : 0.0f;
}

void h3_lora_release_host(h3_lora *lora) {
    if (!lora) return;
    for (size_t index = 0; index < lora->adapter_count; index++) {
        h3_lora_adapter *adapter = &lora->adapters[index];
        free(adapter->a);
        free(adapter->b);
        adapter->a = NULL;
        adapter->b = NULL;
        adapter->have_a = 0;
        adapter->have_b = 0;
    }
}

/* Rows of the fused parent matrix that one adapter row maps onto. */
static int lora_parent_row(const h3_lora_adapter *adapter,
                            const h3_lora_layout *layout, uint32_t row,
                            uint32_t *parent_row) {
    switch (adapter->slot) {
    case H3_LORA_SLOT_FULL:
        *parent_row = row;
        return 1;
    case H3_LORA_SLOT_Q:
    case H3_LORA_SLOT_K:
    case H3_LORA_SLOT_V: {
        if (!layout->head_dim) return 0;
        uint32_t head = row / layout->head_dim;
        uint32_t dimension = row % layout->head_dim;
        uint32_t which = adapter->slot == H3_LORA_SLOT_Q ? 0 :
            (adapter->slot == H3_LORA_SLOT_K ? 1 : 2);
        *parent_row = head * layout->head_dim * 3 + which * layout->head_dim +
                      dimension;
        return 1;
    }
    case H3_LORA_SLOT_GATE:
        *parent_row = row;
        return 1;
    case H3_LORA_SLOT_UP:
        *parent_row = layout->ffn + row;
        return 1;
    }
    return 0;
}

/* Metal releases a buffer that is dropped before its command buffer is
 * committed, which would discard the merged result. The temporaries are
 * therefore released only after the owner reports a completed submit. The
 * list grows on demand so a command buffer may merge any number of
 * matrices. Returns 0 only when the tensor cannot be tracked. */
static int lora_defer(h3_lora *lora, h3_gpu_tensor *tensor) {
    if (!tensor) return 1;
    if (lora->pending_count == lora->pending_capacity) {
        size_t capacity = lora->pending_capacity ?
            lora->pending_capacity * 2 : H3_LORA_PENDING_INITIAL;
        if (capacity > SIZE_MAX / sizeof(*lora->pending)) return 0;
        h3_gpu_tensor **grown = realloc(lora->pending,
                                        capacity * sizeof(*grown));
        if (!grown) return 0;
        lora->pending = grown;
        lora->pending_capacity = capacity;
    }
    lora->pending[lora->pending_count++] = tensor;
    return 1;
}

void h3_lora_flush(h3_lora *lora) {
    if (!lora) return;
    for (size_t index = 0; index < lora->pending_count; index++)
        h3_gpu_tensor_free(lora->pending[index]);
    lora->pending_count = 0;
}

static int lora_merge_one(h3_lora *lora, h3_gpu *gpu,
                           h3_gpu_tensor *weight,
                           const h3_lora_adapter *adapter, uint32_t rows,
                           uint32_t columns, const h3_lora_layout *layout,
                           float scale, char *error, size_t error_size) {
    uint32_t rank = adapter->rank;
    size_t padded_elements = (size_t)rows * rank;
    float *padded = calloc(padded_elements, sizeof(float));
    float *transposed = malloc((size_t)columns * rank * sizeof(float));
    if (!padded || !transposed) {
        free(padded);
        free(transposed);
        fail(error, error_size, "out of memory merging a LoRA adapter");
        return 0;
    }
    for (uint32_t row = 0; row < adapter->rows; row++) {
        uint32_t parent_row = 0;
        if (!lora_parent_row(adapter, layout, row, &parent_row) ||
            parent_row >= rows) {
            free(padded);
            free(transposed);
            fail(error, error_size,
                 "LoRA adapter %s does not fit the target matrix",
                 adapter->scope);
            return 0;
        }
        for (uint32_t index = 0; index < rank; index++) {
            padded[(size_t)parent_row * rank + index] =
                adapter->b[(size_t)row * rank + index];
        }
    }
    for (uint32_t index = 0; index < rank; index++) {
        for (uint32_t column = 0; column < columns; column++) {
            transposed[(size_t)column * rank + index] =
                adapter->a[(size_t)index * columns + column];
        }
    }
    double started = h3_lora_now();
    h3_gpu_tensor *b_tensor = h3_gpu_tensor_from_f32(gpu, padded,
                                                     padded_elements);
    h3_gpu_tensor *a_tensor = h3_gpu_tensor_from_f32(gpu, transposed,
                                                     (size_t)columns * rank);
    int ok = b_tensor && a_tensor &&
             h3_gpu_lora_merge_bf16(gpu, weight, b_tensor, a_tensor, rows,
                                    columns, rank, scale);
    int keep_b = lora_defer(lora, b_tensor);
    int keep_a = lora_defer(lora, a_tensor);
    if (!keep_b || !keep_a) {
        /* The load aborts on this error, so the command buffer never
         * commits and freeing the untracked temporaries is safe. */
        if (!keep_b) h3_gpu_tensor_free(b_tensor);
        if (!keep_a) h3_gpu_tensor_free(a_tensor);
        fail(error, error_size, "cannot track the LoRA merge temporaries");
        return 0;
    }
    free(padded);
    free(transposed);
    if (!ok) {
        fail(error, error_size, "cannot merge the LoRA adapter %s: %s",
             adapter->scope, h3_gpu_error(gpu));
        return 0;
    }
    lora->applied++;
    lora->merge_count++;
    lora->merge_seconds += h3_lora_now() - started;
    return 1;
}

int h3_lora_merge(h3_lora *lora, h3_gpu *gpu, h3_gpu_tensor *weight,
                  const char *scope, h3_lora_module module_kind, uint32_t rows,
                  uint32_t columns, const h3_lora_layout *layout,
                  char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!lora || !lora->adapter_count) return 1;
    if (!gpu || !weight || !scope || !layout) {
        fail(error, error_size, "invalid LoRA merge request");
        return 0;
    }
    if (h3_gpu_tensor_elements(weight) != (size_t)rows * columns) {
        fail(error, error_size,
             "LoRA target %s has %zu elements, expected %zu", scope,
             h3_gpu_tensor_elements(weight), (size_t)rows * columns);
        return 0;
    }
    if ((size_t)rows * columns > UINT32_MAX) {
        fail(error, error_size, "the LoRA target %s is too large to merge",
             scope);
        return 0;
    }
    for (size_t index = 0; index < lora->adapter_count; index++) {
        h3_lora_adapter *adapter = &lora->adapters[index];
        if (strcmp(adapter->scope, scope)) continue;
        if (adapter->module_kind != module_kind) continue;
        if (!adapter->a || !adapter->b || !adapter->rank) continue;
        if (adapter->columns != columns) {
            fprintf(stderr,
                    "h3: skipping LoRA adapter %s: expected %u input "
                    "channels, target has %u\n",
                    adapter->scope, adapter->columns, columns);
            continue;
        }
        uint32_t expected_rows = 0;
        switch (adapter->slot) {
        case H3_LORA_SLOT_FULL:
            expected_rows = rows;
            break;
        case H3_LORA_SLOT_Q:
        case H3_LORA_SLOT_K:
        case H3_LORA_SLOT_V:
            expected_rows = layout->inner;
            break;
        case H3_LORA_SLOT_GATE:
        case H3_LORA_SLOT_UP:
            expected_rows = layout->ffn;
            break;
        }
        if (adapter->rows != expected_rows) {
            fprintf(stderr,
                    "h3: skipping LoRA adapter %s: expected %u output rows, "
                    "target has %u\n",
                    adapter->scope, adapter->rows, expected_rows);
            continue;
        }
        /* Without metadata every adapter scales by one (alpha = rank); with
         * metadata the single alpha applies to every adapter. */
        float alpha = lora->alpha_from_rank ? (float)adapter->rank :
                                              lora->alpha;
        float scale = alpha / (float)adapter->rank * lora->scale;
        if (!isfinite(scale) || scale == 0.0f) continue;
        if (!lora_merge_one(lora, gpu, weight, adapter, rows, columns, layout,
                            scale, error, error_size)) return 0;
    }
    return 1;
}
