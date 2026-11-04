#!/usr/bin/env python3
"""Merge one or more H3 LoRAs into one h3.c-compatible adapter.

Usage: lora_tools.py OUT 'PATH:SCALE' ['PATH:SCALE' ...]

- Accepts standard names (`blocks.N...lora_A/B`, `lora_down/up`,
  `diffusion_model.` prefix) and F16/BF16/F32 tensors.
- Per-LoRA scale and per-scope `.alpha` are baked in; the output uses a
  uniform rank equal to the sum of the inputs' ranks (missing scopes are
  zero-padded), with metadata alpha = rank, so h3.c's alpha/rank = 1.
- `lora_unet_...` (Kohya) spellings are refused for now.
"""
import json
import re
import struct
import sys

import mlx.core as mx

ROLE_RE = re.compile(r"\.(lora_A| lora_B|lora_a|lora_b|lora_down|lora_up|alpha)(?:\.weight)?$".replace(" ", ""))


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def parse(path):
    h = header(path)
    meta = h.get("__metadata__", {})
    a, b, alpha = {}, {}, {}
    for key in h:
        if key == "__metadata__":
            continue
        if key.startswith("lora_unet_"):
            rest = key[len("lora_unet_"):]
            m = re.match(r"(.+)\.(lora_down|lora_up|alpha)(?:\.weight)?$", rest)
            if not m:
                continue
            scope_part, role = m.group(1), m.group(2)
            m2 = re.match(
                r"(token_refiner_)?blocks_(\d+)_(attn_out_proj|attn_qkv_proj|mlp_fc1|mlp_fc2)$",
                scope_part,
            )
            if not m2:
                raise SystemExit(f"unsupported Kohya scope: {key}")
            prefix = "token_refiner." if m2.group(1) else ""
            module = {
                "attn_out_proj": "attn.out_proj",
                "attn_qkv_proj": "attn.qkv_proj",
                "mlp_fc1": "mlp.fc1",
                "mlp_fc2": "mlp.fc2",
            }[m2.group(3)]
            scope = f"{prefix}blocks.{m2.group(2)}.{module}"
            if role == "lora_down":
                a[scope] = key
            elif role == "lora_up":
                b[scope] = key
            else:
                alpha[scope] = key
            continue
        k = key
        for prefix in ("diffusion_model.", "base_model.model.", "model.diffusion_model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        m = ROLE_RE.search(k)
        if not m:
            continue
        scope = k[: m.start()].rstrip(".")
        role = m.group(1).lower()
        if role in ("lora_a", "lora_down"):
            a[scope] = key
        elif role in ("lora_b", "lora_up"):
            b[scope] = key
        elif role == "alpha":
            alpha[scope] = key
    if not a or set(a) != set(b):
        raise SystemExit(f"unpaired or empty adapters: {path}")
    meta_alpha = None
    for k in ("alpha", "network_alpha", "lora_alpha"):
        if k in meta:
            try:
                meta_alpha = float(meta[k])
            except ValueError:
                pass
            break
    return a, b, alpha, meta_alpha


def scalar(value):
    try:
        return float(value.item())
    except Exception:
        return float(mx.max(value))


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out_path = sys.argv[1]
    specs = []
    for arg in sys.argv[2:]:
        path, _, scale = arg.rpartition(":")
        if not path:
            path, scale = arg, "1"
        specs.append((path, float(scale)))
    data = []
    for path, scale in specs:
        a, b, alpha, meta_alpha = parse(path)
        arrays = mx.load(path)
        ranks = {s: arrays[a[s]].shape[0] for s in a}
        data.append((path, scale, a, b, alpha, meta_alpha, arrays, ranks))
    total_rank = sum(max(d[7].values()) for d in data)
    scopes = sorted(set().union(*(set(d[2]) for d in data)))
    out = {}
    for scope in scopes:
        in_dim = out_dim = None
        for d in data:
            if scope in d[2]:
                in_dim = d[6][d[2][scope]].shape[1]
                out_dim = d[6][d[3][scope]].shape[0]
                break
        if in_dim is None:
            raise SystemExit(f"cannot size scope {scope}")
        parts_a, parts_b = [], []
        for path, scale, a, b, alpha, meta_alpha, arrays, ranks in data:
            r = max(ranks.values())
            if scope in a:
                A = arrays[a[scope]].astype(mx.float32)
                B = arrays[b[scope]].astype(mx.float32)
                rank = A.shape[0]
                factor = ((scalar(arrays[alpha[scope]]) if scope in alpha else
                           (meta_alpha if meta_alpha else rank)) / rank) * scale
                A = A * factor
                if rank < r:
                    A = mx.concatenate([A, mx.zeros((r - rank, in_dim), dtype=mx.float32)], axis=0)
                    B = mx.concatenate([B, mx.zeros((out_dim, r - rank), dtype=mx.float32)], axis=1)
            else:
                A = mx.zeros((r, in_dim), dtype=mx.float32)
                B = mx.zeros((out_dim, r), dtype=mx.float32)
            parts_a.append(A)
            parts_b.append(B)
        out[f"{scope}.lora_a"] = mx.concatenate(parts_a, axis=0)
        out[f"{scope}.lora_b"] = mx.concatenate(parts_b, axis=1)
    mx.save_safetensors(out_path, out, metadata={"alpha": str(total_rank), "format": "pt"})
    print(f"wrote {out_path}: {len(scopes)} scopes, rank {total_rank}, "
          f"{len(specs)} loras")


if __name__ == "__main__":
    main()
