# Raw Benchmark Results

Full data from the TurboQuant KV cache experiment (v2 protocol). See [README.md](README.md) for the summary.

**Date:** 2026-03-27
**Machine:** MacBook Pro, Apple M3 Max, 64 GB unified memory
**Build:** b8506-dfc109798 (Metal, Flash Attention enabled)
**Runs:** 3 per config, median reported

---

## Gemma 3 4B Instruct Q4_K_M

### KV Cache Memory (constant across prompt lengths)

| Config | Non-SWA K | Non-SWA V | SWA K | SWA V | **Total** | **Compression** |
|--------|-----------|-----------|-------|-------|-----------|-----------------|
| baseline (f16) | 80.00 MiB | 80.00 MiB | 87.00 MiB | 87.00 MiB | **334 MiB** | 1x |
| q8_0 | 42.50 MiB | 42.50 MiB | 46.22 MiB | 46.22 MiB | **177 MiB** | 1.9x |
| q4_0 | 22.50 MiB | 22.50 MiB | 24.47 MiB | 24.47 MiB | **94 MiB** | 3.6x |
| turbo3 | 17.50 MiB | 17.50 MiB | 19.03 MiB | 19.03 MiB | **73 MiB** | 4.6x |
| turbo4 | 21.25 MiB | 21.25 MiB | 23.11 MiB | 23.11 MiB | **89 MiB** | 3.8x |

### GPU Memory Breakdown

| Config | Model | Context (KV) | Compute | **GPU Total** |
|--------|-------|-------------|---------|---------------|
| baseline | 2368 MiB | 334 MiB | 517 MiB | 3219 MiB |
| q8_0 | 2368 MiB | 177 MiB | 517 MiB | 3062 MiB |
| q4_0 | 2368 MiB | 94 MiB | 517 MiB | 2979 MiB |
| turbo3 | 2368 MiB | 73 MiB | 517 MiB | 2958 MiB |
| turbo4 | 2368 MiB | 89 MiB | 517 MiB | CRASH |

### Generation Speed (tokens/sec) — 3 runs per config

| Prompt | Config | Run 1 | Run 2 | Run 3 | **Median** |
|--------|--------|-------|-------|-------|------------|
| short | baseline | 99.4 | 99.8 | 99.5 | **99.5** |
| short | q8_0 | 89.0 | 89.3 | 88.5 | **89.0** |
| short | q4_0 | 81.6 | 84.7 | 84.7 | **84.7** |
| short | turbo3 | 58.5 | 53.1 | 53.3 | **53.3** |
| medium | baseline | 55.6 | 47.5 | 43.9 | **47.5** |
| medium | q8_0 | 39.8 | 36.8 | 34.3 | **36.8** |
| medium | q4_0 | 32.1 | 32.3 | 34.4 | **32.3** |
| medium | turbo3 | 16.2 | 18.9 | 17.7 | **17.7** |
| long | baseline | 34.3 | 31.6 | 26.7 | **31.6** |
| long | q8_0 | 29.2 | 29.4 | 28.8 | **29.2** |
| long | q4_0 | 39.5 | 33.4 | 29.8 | **33.4** |
| long | turbo3 | 17.7 | 21.8 | 20.0 | **20.0** |

### Prompt Eval Speed (tokens/sec)

| Prompt | Config | Run 1 | Run 2 | Run 3 | **Median** |
|--------|--------|-------|-------|-------|------------|
| short | baseline | 896.4 | 835.9 | 903.2 | **896.4** |
| short | q8_0 | 506.6 | 892.9 | 899.3 | **892.9** |
| short | q4_0 | 434.9 | 895.6 | 898.5 | **895.6** |
| short | turbo3 | 821.2 | 740.0 | 829.1 | **821.2** |
| medium | baseline | 1174.0 | 1042.8 | 927.4 | **1042.8** |
| medium | q8_0 | 1551.7 | 781.1 | 719.8 | **781.1** |
| medium | q4_0 | 1468.4 | 650.1 | 686.1 | **686.1** |
| medium | turbo3 | 1504.1 | 578.6 | 620.6 | **620.6** |
| long | baseline | 646.1 | 819.0 | 838.6 | **819.0** |
| long | q8_0 | 712.7 | 795.6 | 774.7 | **774.7** |
| long | q4_0 | 944.8 | 780.2 | 802.1 | **802.1** |
| long | turbo3 | 661.1 | 763.4 | 661.4 | **661.4** |

> **Note:** Prompt eval times show high variance, especially on short prompts where the total eval time is only 50-100ms. Run 1 is often significantly different from runs 2-3 due to GPU shader compilation and cache warm-up. The median smooths this out, but the individual values reflect real measurement noise.

### Output Quality — Gemma

| Config | short | medium | long |
|--------|-------|--------|------|
| baseline | Coherent, complete | Coherent, complete | Coherent, complete |
| q8_0 | Coherent, complete | Coherent, complete | Coherent, complete |
| q4_0 | Coherent, complete | Coherent, complete | Coherent, complete |
| turbo3 | Garbage from token 1 | Garbage after prompt | Garbage after prompt |
| turbo4 | CRASH | CRASH | CRASH |

---

## Llama 3.1 8B Instruct Q4_K_M

### KV Cache Memory

| Config | K | V | **Total** | **Compression** |
|--------|---|---|-----------|-----------------|
| baseline (f16) | 512 MiB | 512 MiB | **1024 MiB** | 1x |
| q8_0 | 272 MiB | 272 MiB | **544 MiB** | 1.9x |
| q4_0 | 144 MiB | 144 MiB | **288 MiB** | 3.6x |
| turbo3 | — | — | **CRASH** | — |

### GPU Memory Breakdown

| Config | Model | Context (KV) | Compute | **GPU Total** |
|--------|-------|-------------|---------|---------------|
| baseline | 4685 MiB | 1024 MiB | 258 MiB | 5967 MiB |
| q8_0 | 4685 MiB | 544 MiB | 258 MiB | 5487 MiB |
| q4_0 | 4685 MiB | 288 MiB | 258 MiB | 5231 MiB |
| turbo3 | — | — | — | CRASH |

### Generation Speed (tokens/sec) — 3 runs per config

| Prompt | Config | Run 1 | Run 2 | Run 3 | **Median** |
|--------|--------|-------|-------|-------|------------|
| short | baseline | 19.5 | 21.0 | 24.7 | **21.0** |
| short | q8_0 | 22.2 | 24.8 | 25.1 | **24.8** |
| short | q4_0 | 23.5 | 25.3 | 25.0 | **25.0** |
| medium | baseline | 26.0 | 25.4 | 25.6 | **25.6** |
| medium | q8_0 | 25.1 | 23.1 | 23.3 | **23.3** |
| medium | q4_0 | 25.9 | 26.6 | 26.7 | **26.6** |
| long | baseline | 27.0 | 28.5 | 29.2 | **28.5** |
| long | q8_0 | 31.3 | 32.4 | 33.0 | **32.4** |
| long | q4_0 | 34.0 | 34.9 | 33.7 | **34.1** |

### Prompt Eval Speed (tokens/sec)

| Prompt | Config | Run 1 | Run 2 | Run 3 | **Median** |
|--------|--------|-------|-------|-------|------------|
| short | baseline | 91.2 | 447.5 | 482.3 | **447.5** |
| short | q8_0 | 147.4 | 186.1 | 481.0 | **186.1** |
| short | q4_0 | 141.1 | 480.7 | 479.0 | **479.0** |
| medium | baseline | 424.3 | 568.4 | 354.9 | **424.3** |
| medium | q8_0 | 392.3 | 423.7 | 401.3 | **401.3** |
| medium | q4_0 | 689.3 | 426.6 | 424.7 | **426.6** |
| long | baseline | 443.4 | 451.7 | 375.6 | **443.4** |
| long | q8_0 | 513.2 | 481.3 | 483.0 | **483.0** |
| long | q4_0 | 498.2 | 509.4 | 529.9 | **509.4** |

> **Note:** Prompt eval on Llama short prompts (49 tokens) shows extreme Run 1 variance — the first run after model load incurs GPU shader compilation overhead that dominates the ~100ms eval time. This stabilizes on subsequent runs.

### Llama turbo3 Crash Details

All 9 turbo3 runs crashed identically during KV cache initialization:

```
ggml/src/ggml.c:1760: GGML_ASSERT(obj_new) failed
```

Stack trace points to `llama_kv_cache` constructor — the turbo3 block structure fails to allocate for Llama 3.1's architecture (128-dim heads, 8 KV heads).

### Output Quality — Llama

| Config | short | medium | long |
|--------|-------|--------|------|
| baseline | Coherent, complete | Coherent, complete | Coherent, complete |
| q8_0 | Coherent, complete | Coherent, complete | Coherent, complete |
| q4_0 | Coherent, complete | Coherent, complete | Coherent, complete |
| turbo3 | CRASH | CRASH | CRASH |

---

## Exact Commands

All commands are saved in each run's `*_command.txt` file. General format:

```bash
./build/bin/llama-completion \
  -m models/<model>.gguf \
  -ngl 99 -c 8192 -n 512 \
  -f prompts/<prompt>.txt \
  --seed 42 --temp 0 -no-cnv \
  --cache-type-k <type> --cache-type-v <type>
```

Where `<type>` is one of: `f16`, `q8_0`, `q4_0`, `turbo3`, `turbo4`.

---

## Run Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `-ngl 99` | All layers on GPU | Removes CPU/GPU split as variable |
| `-c 8192` | Context window | Large enough to stress KV cache |
| `-n 512` | Tokens to generate | Enough to observe quality degradation |
| `--seed 42` | Fixed seed | Deterministic output for comparison |
| `--temp 0` | Greedy sampling | Removes sampling randomness |
| `-no-cnv` | Completion mode | No chat template, clean output |
| 5s pause | Between runs | Let GPU cool and memory settle |
