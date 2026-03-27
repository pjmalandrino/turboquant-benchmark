# TurboQuant Fork — Code Analysis

A look at the implementation issues I found in [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (branch `feature/turboquant-kv-cache`).

This is not a review of the TurboQuant algorithm itself (which is sound), but of the llama.cpp integration in its current state.

---

## Architecture Overview

TurboQuant adds two new quantization types for the KV cache:

| Type | Bits per value | Compression vs f16 | Target |
|------|---------------|-------------------|--------|
| `turbo3` | 3.25-3.50 | 4.6-4.9x | Aggressive compression |
| `turbo4` | 4.25 | 3.8x | Higher quality variant |

The approach uses a **Walsh-Hadamard Transform (WHT)** rotation to spread information across dimensions before quantization, reducing the impact of aggressive quantization on any single dimension.

### Files modified from upstream llama.cpp

| File | Changes |
|------|---------|
| `ggml/include/ggml.h` | New type enums for turbo3/turbo4 |
| `ggml/src/ggml-common.h` | Block structures |
| `ggml/src/ggml-quants.h` | Function declarations |
| `ggml/src/ggml-turbo-quant.c` | **New file** — quantize/dequantize ops (CPU) |
| `ggml/src/ggml-metal/` | Metal shaders for GPU quantize/dequantize |
| `src/llama-graph.cpp` | Query pre-rotation logic |
| `src/llama-kv-cache.cpp` | KV cache type registration |

---

## Issue 1: Fragile WHT Rotation Coupling

### The design

TurboQuant's key insight: apply a WHT rotation to queries **before** attention, rather than applying an inverse rotation when dequantizing K/V. This is mathematically equivalent (WHT is its own inverse) and avoids the dequant overhead.

### What I found

In `src/llama-graph.cpp`, the query pre-rotation is conditional — it only applies when the head dimension is divisible by 128:

```cpp
if (q->ne[0] % 128 == 0) {
    q = ggml_turbo_wht(ctx0, q, 0);
}
```

Meanwhile, the Metal dequantization shaders have **removed** the inverse rotation entirely, relying on this pre-rotation to always happen.

This creates a fragile coupling: if the pre-rotation is ever skipped (model with non-aligned head dims), Q and K/V end up in different vector spaces, and attention produces garbage.

For Gemma 3 (`n_embd_head_k = 256`), the condition passes — so the rotation does happen. But the output still degrades immediately. The cumulative quantization error at 3.25 bits is too aggressive for this architecture:

- 256-dim heads where small per-element errors add up
- GQA with 4 KV heads, each serving 2 query heads (amplifies error)
- Sliding window attention + global attention = two separate KV caches, both quantized

### Llama 3.1 crash (new finding in v2)

On Llama 3.1 8B (128-dim heads, 8 KV heads), turbo3 doesn't even get to inference — it crashes during KV cache construction with `GGML_ASSERT(obj_new) failed`. The turbo3 block structure appears incompatible with this head dimension/KV head count combination. This happens consistently across all prompt lengths and runs.

---

## Issue 2: CPU Quantization is a Stub

Looking at `ggml/src/ggml-turbo-quant.c`, the CPU quantization function doesn't actually quantize — it zeroes out the output arrays. It's a placeholder.

This means:
- No CPU-only fallback is possible
- If any KV cache computation falls back to CPU for any reason, you get zeros
- Mixed CPU/GPU inference would silently corrupt results

In practice with `-ngl 99` and a model that fits in GPU memory, this path probably isn't hit. But it's worth knowing it's not a safety net.

---

## Issue 3: Missing Turbo4 Metal Kernels

This one is straightforward. The Metal shaders include:

- turbo3: both vectorized **and** non-vectorized Flash Attention kernels
- turbo4: **only** vectorized kernels

The non-vectorized path is used during prompt evaluation (when batch size >= 20). Since no turbo4 non-vectorized kernel exists, Metal can't compile the pipeline and the process segfaults.

The error message is clear:

```
ggml_metal_library_compile_pipeline: failed to compile pipeline:
  'kernel_flash_attn_ext_turbo4_dk256_dv256'
Function not found in the library
```

A possible workaround would be disabling Flash Attention (`--flash-attn off`), but I haven't tested this.

---

## Summary

| Issue | Severity | Type |
|-------|----------|------|
| WHT rotation coupling | High | Design fragility |
| Cumulative quantization error | High | Confirmed on 2 models |
| KV cache init crash (Llama) | Critical | Incompatible with 128-dim heads |
| CPU quantization stub | Medium | Incomplete implementation |
| Missing turbo4 non-vec kernels | Critical | Incomplete implementation |

The fork is clearly a work-in-progress proof of concept. The underlying algorithm is well-motivated, but the llama.cpp integration has gaps. None of these are fundamental flaws — they're all fixable.

**v2 update:** Testing on Llama 3.1 8B revealed that turbo3 is not just a quality issue — it crashes outright on models with 128-dim heads. This narrows the scope of supported architectures significantly.
