/* Style-LoRA support for the H3 DiT transformer.
 *
 * A LoRA checkpoint stores a low-rank update for selected linear layers of
 * the diffusion transformer. The engine merges those updates into the
 * released BF16 weights before the int8 quantization pass, so every later
 * code path (resident weights, SSD streaming, BF16 fallbacks) observes the
 * same adapted model.
 */
#ifndef H3_LORA_H
#define H3_LORA_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_lora h3_lora;

/* The fused projections of one H3 DiT block that a LoRA can adapt. */
typedef enum {
    H3_LORA_MODULE_QKV = 0,
    H3_LORA_MODULE_OUT,
    H3_LORA_MODULE_FC1,
    H3_LORA_MODULE_FC2
} h3_lora_module;

/* Row layout of the fused H3 projections. The attention QKV matrix stores
 * [head][q|k|v][dimension] rows and the MLP input matrix stores the gate
 * rows before the up rows. Split adapters (q, k, v, gate, up) are scattered
 * into those fused matrices with this layout. */
typedef struct {
    uint32_t heads;
    uint32_t head_dim;
    uint32_t inner;
    uint32_t ffn;
} h3_lora_layout;

/* Load a LoRA safetensors file. The scale multiplies the checkpoint alpha
 * and defaults to 1.0. Returns NULL and fills error on failure. */
h3_lora *h3_lora_load(const char *path, float scale,
                       char *error, size_t error_size);
void h3_lora_free(h3_lora *lora);

/* True when the checkpoint contains at least one usable adapter. */
int h3_lora_active(const h3_lora *lora);
size_t h3_lora_adapter_count(const h3_lora *lora);
size_t h3_lora_tensor_count(const h3_lora *lora);
size_t h3_lora_skipped_count(const h3_lora *lora);
size_t h3_lora_applied_count(const h3_lora *lora);
/* Host-side cost of the merges and the number of merged matrices. */
size_t h3_lora_merge_count(const h3_lora *lora);
double h3_lora_merge_seconds(const h3_lora *lora);
/* True when the checkpoint carries at least one adapter for the scope. */
int h3_lora_has_scope(const h3_lora *lora, const char *scope);
const char *h3_lora_path(const h3_lora *lora);
float h3_lora_scale(const h3_lora *lora);
float h3_lora_alpha(const h3_lora *lora);
/* Step-acceleration checkpoints carry the schedule they were trained with.
 * The engine adopts these values when the user does not override them. */
int h3_lora_recommended_steps(const h3_lora *lora);
float h3_lora_recommended_shift(const h3_lora *lora);

/* Release the merge temporaries of the previous merges. Call this after the
 * command buffer that used them has been committed and completed. */
void h3_lora_flush(h3_lora *lora);

/* Drop the host A/B copies. The resident load merges every block up front, so
 * the copies are dead weight afterwards; later merges skip the freed
 * adapters instead of failing. Streamed loads must keep them. */
void h3_lora_release_host(h3_lora *lora);

/* Merge every adapter that targets the given scope into a BF16 weight
 * matrix with the given shape. The scope names the block that owns the
 * projection, for example "blocks.3.attn" or "blocks.3.mlp". The merge is
 * encoded into the caller's Metal command buffer, which must already be
 * open. Adapters whose shapes do not match the target matrix are skipped. */
int h3_lora_merge(h3_lora *lora, h3_gpu *gpu, h3_gpu_tensor *weight,
                  const char *scope, h3_lora_module module, uint32_t rows,
                  uint32_t columns, const h3_lora_layout *layout,
                  char *error, size_t error_size);

#endif
