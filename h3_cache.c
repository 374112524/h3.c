#include "h3_cache.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>

static uint64_t cache_hash_update(uint64_t hash, const void *data, size_t size) {
    const unsigned char *bytes = data;
    for (size_t index = 0; index < size; index++) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t cache_hash_text(uint64_t hash, const char *text) {
    if (!text) text = "";
    return cache_hash_update(hash, text, strlen(text));
}

static uint64_t cache_hash_number(uint64_t hash, long long value) {
    return cache_hash_update(hash, &value, sizeof(value));
}

static int cache_root(const char *base, const char *namespace_text,
                      uint64_t *hash, char *directory, size_t size) {
    if (!directory || !size || !base || !*base || strlen(base) + 32 > size)
        return 0;
    *hash = cache_hash_text(UINT64_C(1469598103934665603), namespace_text);
    return 1;
}

int h3_cache_directory_text(const char *base_env, const char *namespace_text,
                            const char *identity, char *directory,
                            size_t size) {
    if (!base_env || !namespace_text) return 0;
    const char *base = getenv(base_env);
    uint64_t hash = 0;
    if (!cache_root(base, namespace_text, &hash, directory, size)) return 0;
    hash = cache_hash_text(hash, identity);
    snprintf(directory, size, "%s/%016llx", base, (unsigned long long)hash);
    return 1;
}

int h3_cache_directory(const h3_weight_store *store, const char *variant,
                       char *directory, size_t size) {
    const char *base = getenv("H3_INT8_CACHE");
    uint64_t hash = 0;
    if (!cache_root(base, "h3-cache-v1|", &hash, directory, size)) return 0;
    hash = cache_hash_text(hash, variant);
    for (size_t index = 0;; index++) {
        const char *path = h3_weight_store_path(store, index);
        if (!path) break;
        hash = cache_hash_text(hash, path);
        struct stat status;
        if (stat(path, &status) != 0) {
            hash = cache_hash_number(hash, -1);
            continue;
        }
        hash = cache_hash_number(hash, (long long)status.st_size);
        hash = cache_hash_number(hash, (long long)status.st_mtimespec.tv_sec);
        hash = cache_hash_number(hash, (long long)status.st_mtimespec.tv_nsec);
    }
    snprintf(directory, size, "%s/%016llx", base, (unsigned long long)hash);
    return 1;
}

int h3_cache_env_flag(const char *name) {
    const char *value = name ? getenv(name) : NULL;
    return value && (!strcmp(value, "1") || !strcasecmp(value, "yes"));
}

int h3_cache_writes_enabled(void) { return h3_cache_env_flag("H3_INT8_CACHE_WRITE"); }

int h3_cache_prepare(const char *directory, char *error, size_t error_size) {
    if (!directory || !*directory) return 0;
    /* Create every component, not just the leaf. A cache root the caller
     * pointed at but never made should still work; a single mkdir here used to
     * fail with ENOENT and every writer that ignores the error text then
     * silently disabled itself. */
    char path[1024];
    const size_t length = strlen(directory);
    if (length >= sizeof(path)) {
        if (error && error_size)
            snprintf(error, error_size, "cache directory path is too long");
        return 0;
    }
    memcpy(path, directory, length + 1);
    for (size_t index = 1; index <= length; index++) {
        if (path[index] != '/' && path[index] != '\0') continue;
        const char stopped = path[index];
        path[index] = '\0';
        if (mkdir(path, 0777) != 0 && errno != EEXIST) {
            if (error && error_size)
                snprintf(error, error_size, "cannot create cache directory %s: %s",
                         path, strerror(errno));
            return 0;
        }
        path[index] = stopped;
    }
    struct stat status;
    if (stat(directory, &status) == 0 && S_ISDIR(status.st_mode)) return 1;
    if (error && error_size)
        snprintf(error, error_size, "%s is not a directory", directory);
    return 0;
}

int h3_cache_path(const char *directory, const char *name,
                  char *path, size_t size) {
    if (!directory || !name || !path || !size) return 0;
    int written = snprintf(path, size, "%s/%s", directory, name);
    return written > 0 && (size_t)written < size;
}
