# Raw Benchmark Results

Full data from the TurboQuant KV cache experiment. See [README.md](README.md) for the summary and conclusions.

**Date:** 2026-03-26
**Machine:** MacBook Pro, Apple M3 Max, 64 GB unified memory
**Model:** Gemma 3 4B Instruct Q4_K_M (bartowski, 2.31 GiB)
**Build:** b8506-dfc109798 (Metal, Flash Attention enabled)

---

## Exact Commands

### Baseline (f16 KV cache)

```bash
./build/bin/llama-completion \
  -m models/google_gemma-3-4b-it-Q4_K_M.gguf \
  -ngl 99 -c 8192 -n 512 \
  -f prompt.txt \
  --seed 42 --temp 0 -no-cnv \
  --cache-type-k f16 --cache-type-v f16
```

### Turbo3 (3.25 bpv KV cache)

```bash
./build/bin/llama-completion \
  -m models/google_gemma-3-4b-it-Q4_K_M.gguf \
  -ngl 99 -c 8192 -n 512 \
  -f prompt.txt \
  --seed 42 --temp 0 -no-cnv \
  --cache-type-k turbo3 --cache-type-v turbo3
```

### Turbo4 (4.25 bpv KV cache)

```bash
./build/bin/llama-completion \
  -m models/google_gemma-3-4b-it-Q4_K_M.gguf \
  -ngl 99 -c 8192 -n 512 \
  -f prompt.txt \
  --seed 42 --temp 0 -no-cnv \
  --cache-type-k turbo4 --cache-type-v turbo4
```

---

## Raw Metrics (from llama.cpp output)

### Baseline

```
llama_kv_cache: size =  160.00 MiB (  8192 cells,   5 layers), K (f16):   80.00 MiB, V (f16):   80.00 MiB
llama_kv_cache: size =  174.00 MiB (  1536 cells,  29 layers), K (f16):   87.00 MiB, V (f16):   87.00 MiB

common_perf_print:    sampling time =      78.24 ms
common_perf_print:        load time =     243.66 ms
common_perf_print: prompt eval time =     317.86 ms /   500 tokens (    0.64 ms per token,  1573.03 tokens per second)
common_perf_print:        eval time =    5406.22 ms /   511 runs   (   10.58 ms per token,    94.52 tokens per second)
common_perf_print:       total time =    5809.10 ms /  1011 tokens

llama_memory_breakdown_print: |   - MTL0 (Apple M3 Max) | 49152 = 45932 + (3219 =  2368 +     334 +     517) |
llama_memory_breakdown_print: |   - Host                |                   554 =   525 +       0 +      29   |
```

### Turbo3

```
llama_kv_cache: TurboQuant rotation matrices initialized (128x128)
llama_kv_cache: size =   35.00 MiB (  8192 cells,   5 layers), K (turbo3):   17.50 MiB, V (turbo3):   17.50 MiB
llama_kv_cache: TurboQuant rotation matrices initialized (128x128)
llama_kv_cache: size =   38.06 MiB (  1536 cells,  29 layers), K (turbo3):   19.03 MiB, V (turbo3):   19.03 MiB

common_perf_print:    sampling time =      89.92 ms
common_perf_print:        load time =     569.33 ms
common_perf_print: prompt eval time =     371.20 ms /   500 tokens (    0.74 ms per token,  1346.98 tokens per second)
common_perf_print:        eval time =    8817.74 ms /   511 runs   (   17.26 ms per token,    57.95 tokens per second)
common_perf_print:       total time =    9287.90 ms /  1011 tokens

llama_memory_breakdown_print: |   - MTL0 (Apple M3 Max) | 49152 = 46192 + (2958 =  2368 +      73 +     517) |
llama_memory_breakdown_print: |   - Host                |                   554 =   525 +       0 +      29   |
```

### Turbo4

```
llama_kv_cache: TurboQuant rotation matrices initialized (128x128)
llama_kv_cache: size =   42.50 MiB (  8192 cells,   5 layers), K (turbo4):   21.25 MiB, V (turbo4):   21.25 MiB
llama_kv_cache: TurboQuant rotation matrices initialized (128x128)
llama_kv_cache: size =   46.22 MiB (  1536 cells,  29 layers), K (turbo4):   23.11 MiB, V (turbo4):   23.11 MiB

ggml_metal_library_compile_pipeline: failed to compile pipeline:
  base = 'kernel_flash_attn_ext_turbo4_dk256_dv256'
Error: Function kernel_flash_attn_ext_turbo4_dk256_dv256 was not found in the library

Exit code: 139 (SIGSEGV)
```

---

## Output Samples

### Baseline — coherent output (first 300 words)

```
## Technical Architecture: Real-Time Collaborative Document Editor

Here's a proposed architecture for the real-time collaborative document editor,
addressing the specified requirements.

**1. Data Layer: CRDTs and Document Model**

* **CRDT Choice: Yjs** - While Automerge and Diamond Types are viable, Yjs offers
  the best balance of features, maturity, and community support for this complex
  use case.
    * **Operation-Based vs. State-Based:** Operation-based CRDTs (like Yjs) are
      generally better suited for collaborative editing due to their ability to
      handle complex, concurrent edits more efficiently.
* **Document Model:** Yjs uses a "Graph CRDT" which represents the document as a
  directed graph of nodes and edges.

**2. Networking: Real-time Sync Protocol**

* **Transport Layer: WebTransport** - WebTransport offers significant advantages
  over WebSockets for this application.
    * **Reliability:** WebTransport provides reliable, bidirectional streams.
    * **Flow Control:** WebTransport's flow control prevents overwhelming the
      network with too many operations.
    * **Multiplexing:** WebTransport allows multiple streams over a single connection.
```

### Turbo3 — point of degradation

The output is coherent up to approximately token 250, then breaks:

```
[...] Be specific about technology choices, justify each decision, and discuss
failure modes for each component. cáiuna sway bitterly ఉండే му জানিয়েছে
fashionable got‍♂ద్ pe everythingবারের talaga bromineelm教えて dostępneされている
ܕCellStyle நபி oba survivesមី എല്ലാംvived Gotз affirmeAndUpdateäck değer
enlightenment उर्दूہورと同 যেদিন claws➳ ritelumat রম gül蕩 [...]
```

---

## Notes

- Full outputs are in `results/*.txt`
- All runs used the same prompt (`prompt.txt`), same seed (42), same temperature (0)
- `llama-cli` enters chat mode with Gemma 3 — use `llama-completion -no-cnv` instead
