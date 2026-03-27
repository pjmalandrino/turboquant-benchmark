# TurboQuant Fork — Code Analysis

Deep dive into the implementation issues found in [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (branch `feature/turboquant-kv-cache`, build b8506-dfc109798).

This is not a review of the TurboQuant algorithm itself (which is sound), but of the llama.cpp integration.

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
| `ggml/include/ggml.h` | New type enums (`GGML_TYPE_TURBO3_0`, `GGML_TYPE_TURBO4_0`) |
| `ggml/src/ggml-common.h` | Block structures for turbo3/turbo4 |
| `ggml/src/ggml-quants.h` | Function declarations |
| `ggml/src/ggml-turbo-quant.c` | **New file** — quantize/dequantize operations (CPU) |
| `ggml/src/ggml-metal/` | Metal shaders for GPU quantize/dequantize |
| `src/llama-graph.cpp` | Query pre-rotation logic |
| `src/llama-kv-cache.cpp` | KV cache type registration |

---

## Issue 1: Fragile WHT Rotation Coupling

### The design

TurboQuant's key insight: apply a WHT rotation to queries **before** attention, rather than applying an inverse rotation when dequantizing K/V. This is mathematically equivalent (WHT is its own inverse) and avoids the dequant overhead.

### The implementation

In `src/llama-graph.cpp` (~line 2074):

```cpp
// Pre-rotate queries for TurboQuant KV cache
if (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0) {
    if (q->ne[0] % 128 == 0) {
        q = ggml_turbo_wht(ctx0, q, 0);
    }
}
```

In the Metal dequantization shaders:

```metal
// NOTE: turbo_rotate_inverse REMOVED
// Pre-rotate-queries optimization handles this instead
```

### The problem

The rotation is **conditional** (`q->ne[0] % 128 == 0`) but the inverse rotation in dequant is **unconditionally removed**. This creates an implicit contract:

- If pre-rotation **happens**: Q is rotated, K/V are stored rotated -> dot product is correct
- If pre-rotation **doesn't happen**: Q is unrotated, K/V are stored rotated -> dot product is in wrong space -> garbage

For Gemma 3 (`n_embd_head_k = 256`, divisible by 128), the condition passes. But this is fragile — any model with head dimensions not divisible by 128 would silently produce wrong results.

### Why the output still degrades

Even when the rotation is correctly applied, turbo3 at 3.25 bits introduces quantization error on each KV entry. With Gemma 3's architecture:

- **256-dim heads**: large vectors where small per-element errors compound
- **GQA with 4 KV heads**: each KV head serves 2 query heads, amplifying any error
- **Sliding window attention (1024) + global attention**: two separate KV caches, both quantized

The cumulative error grows with sequence length. After ~200-250 tokens, the attention distribution becomes noisy enough that the model attends to wrong positions, producing garbage.

---

## Issue 2: CPU Quantization Stub

### Location

`ggml/src/ggml-turbo-quant.c`, lines ~175-186:

```c
void quantize_row_turbo3_0_ref(const float * restrict x,
                                block_turbo3_0 * restrict y,
                                int64_t k) {
    // ... setup ...
    for (int i = 0; i < nb; i++) {
        // Placeholder: zeros out qs[] and signs[]
        memset(y[i].qs, 0, sizeof(y[i].qs));
        memset(y[i].signs, 0, sizeof(y[i].signs));
        y[i].d = 0.0f;
    }
}
```

### Impact

If any KV cache quantization falls through to the CPU path (e.g., during graph scheduling edge cases, or if Metal buffers are full), the quantized values are **all zeros**. Dequantizing zeros produces noise (due to the scale factor and rotation), corrupting the attention computation.

In practice, with `-ngl 99` and a model that fits entirely in GPU memory, this path is unlikely to be hit. But it means:

- No CPU-only fallback is possible
- Mixed CPU/GPU inference would be broken
- The unit test (`tq_test.c`) may pass with separate test data but doesn't validate the actual inference path

---

## Issue 3: Missing Turbo4 Metal Kernels

### Location

Metal shader file for Flash Attention (in `ggml/src/ggml-metal/`):

### The gap

| Kernel pattern | turbo3 | turbo4 |
|---------------|--------|--------|
| `kernel_flash_attn_ext_turbo3_*` (non-vectorized) | 9 variants | - |
| `kernel_flash_attn_ext_turbo4_*` (non-vectorized) | - | **0 variants** |
| `kernel_flash_attn_ext_vec_turbo3_*` (vectorized) | Defined | - |
| `kernel_flash_attn_ext_vec_turbo4_*` (vectorized) | - | 6 variants |

### Kernel selection logic

In `ggml-metal-ops.cpp` (~line 2530):

```cpp
// Simplified selection logic
bool use_vec = (ne01 < 20) && (ne00 % 32 == 0);
```

- `ne01` = batch size during attention
- During **prompt evaluation**: batch is large (>= 20) -> non-vectorized kernel selected
- During **token generation**: batch = 1 -> vectorized kernel selected

### The crash

When turbo4 is used:
1. Model loads successfully, KV cache is allocated with turbo4 type
2. Prompt evaluation begins, batch size >= 20
3. Non-vectorized turbo4 kernel is requested
4. Metal can't find `kernel_flash_attn_ext_turbo4_dk256_dv256`
5. Pipeline compilation fails
6. **Segfault** (exit code 139)

The error message is clear:

```
ggml_metal_library_compile_pipeline: failed to compile pipeline:
  base = 'kernel_flash_attn_ext_turbo4_dk256_dv256'
Error Domain=MTLLibraryErrorDomain Code=5
  "Function kernel_flash_attn_ext_turbo4_dk256_dv256 was not found in the library"
```

### Workaround (not tested)

In theory, disabling Flash Attention (`--flash-attn off` or `--no-flash-attn`) might bypass this issue by using a different attention path. However, this would likely have significant performance implications.

---

## Summary

| Issue | Severity | Type | Fixable? |
|-------|----------|------|----------|
| WHT rotation coupling | High | Design fragility | Yes — add dimension check or restore inverse rotation fallback |
| Cumulative quantization error (turbo3) | High | Algorithm limitation | Partially — may need per-model tuning or higher bit count |
| CPU quantization stub | Medium | Incomplete implementation | Yes — implement actual quantization |
| Missing turbo4 non-vec kernels | Critical | Incomplete implementation | Yes — add the Metal shader variants |

The fork is clearly a **work-in-progress proof of concept**, not production code. The underlying algorithm (WHT-based KV cache quantization) is well-motivated, but the llama.cpp integration has significant gaps.
