# TurboQuant KV Cache Benchmark

**An honest, reproducible benchmark of [TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) KV cache quantization on Apple Silicon.**

TurboQuant promises up to 4.6x KV cache compression with minimal quality loss. I tested it on two models, three prompt lengths, with standard KV quantization (q8_0, q4_0) as reference. Here's what I found.

---

## TL;DR

**The compression is real, but the quality isn't there yet.**

- turbo3 **produces garbage output** on Gemma 3 4B (all prompt lengths)
- turbo3 **crashes** on Llama 3.1 8B (GGML_ASSERT failure during KV cache init)
- turbo4 **crashes** on both models (missing Metal shader)
- Standard q4_0 already gives **3.6x compression** with zero quality loss and comparable speed

> This is early-stage experimental code, not a production feature. The underlying math is sound — the implementation has gaps.

---

## Test Environment

| | |
|---|---|
| **Machine** | MacBook Pro, Apple M3 Max, 64 GB unified memory |
| **OS** | macOS (Apple Silicon, arm64) |
| **Fork** | [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) branch `feature/turboquant-kv-cache` |
| **Build** | b8506-dfc109798, Metal + Flash Attention (embedded library) |
| **Backend** | GPU (Metal), all layers offloaded |

### Models

| Model | Size | Architecture | Head dim | KV heads |
|-------|------|-------------|----------|----------|
| [Gemma 3 4B Instruct Q4_K_M](https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF) | 2.3 GiB | GQA, sliding window + global | 256 | 4 |
| [Llama 3.1 8B Instruct Q4_K_M](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) | 4.6 GiB | GQA, standard attention | 128 | 8 |

### Protocol

- **5 configs**: baseline (f16), q8_0, q4_0, turbo3, turbo4
- **3 prompt lengths**: short (~50 tokens), medium (~500 tokens), long (~1500 tokens)
- **3 runs per config** — median values reported
- Fixed seed (42), greedy sampling (temp 0), context 8192, 512 tokens generated

See [`PROTOCOL.md`](PROTOCOL.md) for the full test protocol.

---

## Results: Gemma 3 4B

### KV Cache Memory

```
Config      KV cache (total)    Compression    Status
─────────   ────────────────    ───────────    ──────
baseline    334 MiB             1x             OK
q8_0        177 MiB             1.9x           OK
q4_0         94 MiB             3.6x           OK
turbo3       73 MiB             4.6x           Garbage output
turbo4       89 MiB             3.8x           Crash (missing Metal kernel)
```

### Generation Speed (tokens/sec, median of 3 runs)

| Prompt | baseline | q8_0 | q4_0 | turbo3 |
|--------|----------|------|------|--------|
| short | 99.5 | 89.0 | 84.7 | 53.3 |
| medium | 47.5 | 36.8 | 32.3 | 17.7 |
| long | 31.6 | 29.2 | 33.4 | 20.0 |

### Prompt Eval Speed (tokens/sec, median of 3 runs)

| Prompt | baseline | q8_0 | q4_0 | turbo3 |
|--------|----------|------|------|--------|
| short | 896 | 893 | 896 | 821 |
| medium | 1043 | 781 | 686 | 621 |
| long | 819 | 775 | 802 | 661 |

### Output Quality

| Config | short | medium | long |
|--------|-------|--------|------|
| baseline | OK | OK | OK |
| q8_0 | OK | OK | OK |
| q4_0 | OK | OK | OK |
| turbo3 | **Garbage** | **Garbage** | **Garbage** |

turbo3 output degenerates into random multilingual Unicode immediately after the prompt echo:

```
[...] Be specific about technology choices, justify each decision, and discuss
failure modes for each component. cáiuna sway bitterly ఉండే му জানিয়েছে
fashionable got‍♂ద్ pe everythingবারের talaga bromineelm教えて dostępneされている
ܕCellStyle நபி oba survivesមី എല്ലാംvived Gotз affirmeAndUpdateäck değer [...]
```

---

## Results: Llama 3.1 8B

### KV Cache Memory

```
Config      KV cache (total)    Compression    Status
─────────   ────────────────    ───────────    ──────
baseline    1024 MiB            1x             OK
q8_0         544 MiB            1.9x           OK
q4_0         288 MiB            3.6x           OK
turbo3       —                  —              CRASH (GGML_ASSERT)
turbo4       —                  —              CRASH (missing Metal kernel)
```

### Generation Speed (tokens/sec, median of 3 runs)

| Prompt | baseline | q8_0 | q4_0 |
|--------|----------|------|------|
| short | 21.0 | 24.8 | 25.0 |
| medium | 25.6 | 23.3 | 26.6 |
| long | 28.5 | 32.4 | 34.1 |

turbo3 crashes during KV cache initialization with `GGML_ASSERT(obj_new) failed` — the turbo3 block structure is incompatible with Llama 3.1's 128-dim heads.

### Output Quality

| Config | short | medium | long |
|--------|-------|--------|------|
| baseline | OK | OK | OK |
| q8_0 | OK | OK | OK |
| q4_0 | OK | OK | OK |
| turbo3 | **CRASH** | **CRASH** | **CRASH** |

---

## The Real Competitor: q4_0

The most useful finding isn't about TurboQuant — it's that **standard q4_0 KV cache quantization already works well**:

| | q4_0 | turbo3 |
|---|---|---|
| Compression | 3.6x | 4.6x |
| Speed impact | Minimal | -39% to -56% |
| Quality loss | None observed | Total degradation |
| Stability | Solid | Crash or garbage |

q4_0 gives 3.6x compression with zero quality loss and no speed penalty. TurboQuant's extra 1x of compression comes at a steep cost.

---

## Code Analysis

I looked at the fork's source to understand the failures. Three implementation issues:

**1. Fragile WHT rotation coupling** — The query pre-rotation (Walsh-Hadamard Transform) is conditionally applied, but the Metal dequant shaders assume it always happens. Skipping it puts Q and K/V in different vector spaces.

**2. CPU quantization is a stub** — The CPU fallback zeroes out values instead of quantizing. No CPU-only path works.

**3. Missing turbo4 Metal kernels** — Non-vectorized Flash Attention kernels for turbo4 don't exist. Prompt eval (batch > 20) needs them, so it crashes.

See [`CODE_ANALYSIS.md`](CODE_ANALYSIS.md) for details.

---

## Reproduce It Yourself

### Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- cmake, git, Xcode CLI tools
- ~10 GB free disk space (models + build)

### Setup

```bash
git clone https://github.com/pjmalandrino/turboquant-benchmark.git
cd turboquant-benchmark

# Clone and build the TurboQuant fork
git clone -b feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant.git llama-cpp
cd llama-cpp
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# Download models
mkdir -p models
huggingface-cli download bartowski/google_gemma-3-4b-it-GGUF \
  google_gemma-3-4b-it-Q4_K_M.gguf --local-dir models/
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models/

# Copy benchmark files
cp ../run_experiment.sh .
cp -r ../prompts .
```

### Run

```bash
# Single config
./run_experiment.sh baseline -r 3 -p medium -m models/google_gemma-3-4b-it-Q4_K_M.gguf

# Full protocol (all configs, one model)
for config in baseline q8_0 q4_0 turbo3; do
  for prompt in short medium long; do
    ./run_experiment.sh "$config" -r 3 -p "$prompt" -m models/google_gemma-3-4b-it-Q4_K_M.gguf
  done
done
```

### Script usage

```
./run_experiment.sh <config> [-r NUM_RUNS] [-p short|medium|long] [-m MODEL_PATH]

Configs: baseline, q8_0, q4_0, turbo3, turbo4
```

---

## Conclusion

TurboQuant's KV cache compression ratios are real and impressive (4.6x). The underlying Walsh-Hadamard Transform approach is mathematically sound.

But in practice, on the two models I tested:
- Output quality is **broken** (Gemma) or the process **crashes** (Llama)
- Generation speed is **significantly slower**
- Standard q4_0 already provides 3.6x compression with no downsides

This is early-stage experimental code. The issues are implementation gaps (stub CPU path, missing shaders, fragile conditionals), not fundamental algorithm flaws. Worth watching for future updates.

### What I didn't test

- Other model architectures (Phi, Mistral, Qwen...)
- Disabling Flash Attention as a turbo4 workaround
- Larger context windows where KV compression matters more
- The Python prototype ([turboquant_plus](https://github.com/TheTom/turboquant_plus)) for validating the algorithm in isolation

---

## Files

```
.
├── README.md              # This file
├── PROTOCOL.md            # Test protocol (v2)
├── CODE_ANALYSIS.md       # Analysis of the fork's code
├── RESULTS.md             # Full raw benchmark data
├── run_experiment.sh      # Benchmark script
├── prompts/               # short.txt, medium.txt, long.txt
│   ├── short.txt
│   ├── medium.txt
│   └── long.txt
├── results/               # Raw outputs organized by model/prompt
└── LICENSE                # MIT
```

## License

MIT. See [LICENSE](LICENSE).
