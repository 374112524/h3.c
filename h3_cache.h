/* Opt-in disk cache for derived DiT state: the pre-quantized block weights
 * and the materialized AdaLN tables.
 *
 * The BF16 checkpoint is mapped rather than copied, so every cold load faults
 * roughly 62 GiB through the GPU. Persisting the derived tensors removes that
 * traffic without changing a single byte of what the kernels consume, which
 * keeps the default path bit-identical: the cache only skips work, it never
 * alters it.
 *
 * Enable with H3_INT8_CACHE=<directory>. The parent directory must exist; the
 * per-variant subdirectory is created on demand. The key hashes the checkpoint
 * identity (shard path, size, mtime) plus a caller-supplied variant string, so
 * a replaced checkpoint, a different LoRA or a different schedule never reuses
 * a stale entry. */
#ifndef H3_CACHE_H
#define H3_CACHE_H

#include "h3_weights.h"

#include <stddef.h>

/* Returns 0 when the cache is disabled (no H3_INT8_CACHE). Otherwise fills
 * directory with <base>/<16 hex digits> and returns 1. The directory may not
 * exist yet. */
int h3_cache_directory(const h3_weight_store *store, const char *variant,
                       char *directory, size_t size);

/* Populating the cache is opt-in on top of enabling it: H3_INT8_CACHE alone
 * only reads, so no run silently writes ~18 GiB. Set H3_INT8_CACHE_WRITE=1 to
 * let a run fill missing entries. */
int h3_cache_writes_enabled(void);

/* Returns 1 when the named environment variable is set to 1 or yes. */
int h3_cache_env_flag(const char *name);

/* The same directory scheme for a cache whose identity is not the DiT shards -
 * for example the encoder outputs, which depend on the prompt, the reference
 * media and the canvas rather than on quantized weights. Reads the base
 * directory from the named environment variable, so an unset variable simply
 * disables that cache. */
int h3_cache_directory_text(const char *base_env, const char *namespace_text,
                            const char *identity, char *directory, size_t size);

/* Creates directory, including the leaf, if needed. Writes a one-line note to
 * stderr and returns 0 when the location is unusable. */
int h3_cache_prepare(const char *directory, char *error, size_t error_size);

/* Joins directory and name. Returns 0 when the result does not fit. */
int h3_cache_path(const char *directory, const char *name,
                  char *path, size_t size);

#endif
