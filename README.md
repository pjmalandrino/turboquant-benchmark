# TurboQuant KV Cache Benchmark

**An honest, reproducible benchmark of [TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) KV cache quantization on Apple Silicon.**

TurboQuant promises up to 4.6x KV cache compression with minimal quality loss. I tested it on Gemma 3 4B. Here's what I found.

---

## TL;DR

| | Baseline (f16) | Turbo3 (3.25 bpv) | Turbo4 (4.25 bpv) |
|---|---|---|---|
| **KV cache** | 334 MiB | **73 MiB (-78%)** | **89 MiB (-73%)** |
| **Compression** | 1x | **4.6x** | **3.8x** |
| **Generation speed** | 94.5 t/s | 58.0 t/s (-39%) | N/A |
| **Output quality** | Coherent | Garbage after ~250 tokens | N/A |
| **Status** | OK | Broken on this model | Crash (missing Metal kernel) |

> The memory compression is real. Quality issues on Gemma 3 4B — might be model-specific.

---

## Test Environment

| | |
|---|---|
| **Machine** | MacBook Pro, Apple M3 Max, 64 GB unified memory |
| **OS** | macOS (Apple Silicon, arm64) |
| **Model** | [Gemma 3 4B Instruct Q4_K_M](https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF) (2.31 GiB) |
| **Fork** | [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) branch `feature/turboquant-kv-cache` |
| **Build** | b8506-dfc109798, Metal + Flash Attention (embedded library) |
| **Backend** | GPU (Metal), all 35 layers offloaded |

---

## Results in Detail

### Memory: KV Cache Compression

This is where TurboQuant delivers. The compression ratios match the claims.

```
                    Baseline (f16)          Turbo3              Turbo4
                    ──────────────          ──────              ──────
Non-SWA cache       160.00 MiB              35.00 MiB           42.50 MiB
  K                  80.00 MiB (f16)        17.50 MiB (turbo3)  21.25 MiB (turbo4)
  V                  80.00 MiB (f16)        17.50 MiB (turbo3)  21.25 MiB (turbo4)

SWA cache           174.00 MiB              38.06 MiB           46.22 MiB
  K                  87.00 MiB (f16)        19.03 MiB (turbo3)  23.11 MiB (turbo4)
  V                  87.00 MiB (f16)        19.03 MiB (turbo3)  23.11 MiB (turbo4)

Total KV            334 MiB                 73 MiB              89 MiB
Compression         1x                      4.6x                3.8x
GPU total           3219 MiB                2958 MiB            2974 MiB
```

### Performance

| Metric | Baseline (f16) | Turbo3 | Delta |
|--------|---------------|--------|-------|
| Prompt eval | 1573 t/s (0.64 ms/t) | 1347 t/s (0.74 ms/t) | **-14%** |
| Generation | 94.52 t/s (10.58 ms/t) | 57.95 t/s (17.26 ms/t) | **-39%** |
| Total time | 5809 ms | 9288 ms | **+60%** |
| Load time | 244 ms | 569 ms | +133% |

Generation speed drops significantly. The quantize/dequantize overhead and Walsh-Hadamard transform rotation add compute cost that isn't offset by the reduced memory bandwidth.

### Output Quality

**Baseline** produces a coherent, well-structured technical architecture document.

**Turbo3** starts coherent (same structure, similar content) then **degenerates into random Unicode garbage** after approximately 200-250 generated tokens:

```
[...] Be specific about technology choices, justify each decision, and discuss
failure modes for each component. cáiuna sway bitterly ఉండే му জানিয়েছে
fashionable got‍♂ద్ pe everythingবারের talaga bromineelm教えて dostępneされている
ܕCellStyle நபி oba survivesមី എല്ലാംvived Gotз affirmeAndUpdateäck değer
enlightenment उर्दूہورと同 যেদিন claws➳ ritelumat রম gül蕩 [...]
```

Full outputs are saved in [`results/`](results/).

---

## Root Cause Analysis

I dug into the fork's code to understand why. Three issues found:

### 1. Turbo3: Fragile rotation coupling

In `src/llama-graph.cpp`, the query pre-rotation (Walsh-Hadamard Transform) is conditionally applied based on head dimension alignment:

```cpp
if (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0) {
    if (q->ne[0] % 128 == 0) {  // conditional!
        q = ggml_turbo_wht(ctx0, q, 0);
    }
}
```

Meanwhile, the Metal dequantization shaders **removed** the inverse rotation, assuming pre-rotation always happens. If it's ever skipped, Q and K/V end up in different vector spaces, and attention produces garbage.

For Gemma 3 (`n_embd_head_k = 256`), the condition passes. But the cumulative quantization error at 3.25 bits compounds over the sequence, causing attention to diverge after ~250 tokens.

### 2. Turbo3: CPU quantization is a stub

The CPU fallback in `ggml/src/ggml-turbo-quant.c` zeros out the quantized values instead of performing actual quantization. If any KV computation falls back to CPU, the stored values are zeros.

### 3. Turbo4: Missing Metal shader

The non-vectorized Flash Attention kernel for turbo4 (`kernel_flash_attn_ext_turbo4_dk256_dv256`) simply doesn't exist in the Metal shader file. During prompt evaluation (batch > 20), the non-vectorized path is selected, the pipeline compilation fails, and the process segfaults (exit code 139).

See [`CODE_ANALYSIS.md`](CODE_ANALYSIS.md) for the full analysis with code references.

---

## Reproduce It Yourself

### Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- cmake, git, Xcode CLI tools
- ~6 GB free disk space

### Setup

```bash
# Clone this benchmark repo
git clone https://github.com/pjmalandrino/turboquant-benchmark.git
cd turboquant-benchmark

# Clone and build the TurboQuant fork
git clone -b feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant.git llama-cpp
cd llama-cpp
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# Download the model (~2.3 GB)
mkdir -p models
huggingface-cli download bartowski/google_gemma-3-4b-it-GGUF \
  google_gemma-3-4b-it-Q4_K_M.gguf --local-dir models/

# Copy benchmark files into the build directory
cp ../run_experiment.sh ../prompt.txt .
```

### Run

```bash
# Run each config (one at a time)
./run_experiment.sh baseline
./run_experiment.sh turbo3
./run_experiment.sh turbo4    # will crash — expected

# Results are in results/
```

### Run parameters

| Parameter | Value | Flag |
|-----------|-------|------|
| GPU layers | 99 (all offloaded) | `-ngl 99` |
| Context window | 8192 | `-c 8192` |
| Tokens to generate | 512 | `-n 512` |
| Seed | 42 (deterministic) | `--seed 42` |
| Temperature | 0 (greedy) | `--temp 0` |
| Mode | Completion (no chat) | `-no-cnv` |

---

## Conclusion

TurboQuant is a **promising research direction** with genuinely impressive compression ratios. The 4.6x KV cache reduction would be very valuable for memory-constrained deployments.

However, in my testing (Gemma 3 4B, single model):
- Output quality **broke down** with turbo3 after ~250 tokens — this might be model-specific
- turbo4 **crashed** due to incomplete Metal shader coverage
- Generation speed was **39% slower**, not faster
- The CPU fallback is a placeholder (stub that zeros out values)

**This is not a criticism of the project** — it's clearly experimental, work-in-progress code. The underlying math (Walsh-Hadamard Transform based quantization) is sound. The issues are implementation gaps, not fundamental flaws.

### Limitations of this benchmark

- **Single model tested** — Gemma 3 4B may be particularly sensitive to KV quantization (256-dim heads, GQA, sliding window). Results could differ on other architectures.
- **Single run per config** — no variance measured yet. See [`PROTOCOL.md`](PROTOCOL.md) for the improved test plan.
- **No comparison with q8_0/q4_0** — llama.cpp already supports KV cache quantization with standard types. A proper benchmark should include those as reference points.

Worth watching for future updates, especially:
- Complete Metal shader coverage
- Proper CPU fallback
- Quality validation across model architectures

---

## Files

```
.
├── README.md                           # This file
├── PROTOCOL.md                         # Improved test protocol (v2)
├── CODE_ANALYSIS.md                    # Deep dive into the fork's code
├── RESULTS.md                          # Full benchmark data with all metrics
├── run_experiment.sh                   # Benchmark script (bash, configurable)
├── prompt.txt                          # The prompt used for all runs
├── results/                            # Raw outputs from each run
└── LICENSE                             # MIT
```

---

## License

MIT. See [LICENSE](LICENSE).
