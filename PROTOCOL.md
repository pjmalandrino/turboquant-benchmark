# Test Protocol v2

Improved benchmark protocol for TurboQuant KV cache evaluation. The initial results (v1) used a single run on a single model with a single prompt length. This protocol addresses those gaps.

---

## What we're measuring

1. **Memory** — KV cache size reduction (from llama.cpp logs)
2. **Speed** — prompt eval and generation throughput (tokens/sec)
3. **Quality** — does the output stay coherent? At what sequence length does it break?
4. **Stability** — variance across runs, crash/error behavior

---

## Configs to test

| Config | KV cache type | Purpose |
|--------|--------------|---------|
| `baseline` | f16 | Reference (no quantization) |
| `q8_0` | q8_0 | Standard llama.cpp KV quant (2x compression) |
| `q4_0` | q4_0 | Standard llama.cpp KV quant (4x compression) |
| `turbo3` | turbo3 | TurboQuant aggressive (4.6x) |
| `turbo4` | turbo4 | TurboQuant quality (3.8x) |

Adding q8_0 and q4_0 gives context — if q4_0 already gives 4x compression with acceptable quality, TurboQuant needs to do better to justify the complexity.

---

## Models to test

Test at least 2 models with different architectures to avoid drawing conclusions from a single data point:

| Model | Architecture | Why |
|-------|-------------|-----|
| Gemma 3 4B Instruct | GQA, 256-dim heads, sliding window + global | Already tested in v1, keeps results comparable |
| Llama 3.1 8B Instruct | GQA, 128-dim heads, standard attention | Different head size, no sliding window, widely used |

The head dimension matters — TurboQuant's WHT rotation block size is 128, so models with 128-dim heads may behave differently than 256-dim.

---

## Prompts — 3 lengths

Use the same topic across lengths (keeps quality comparison fair):

| Label | Target tokens | Description |
|-------|--------------|-------------|
| `short` | ~100 tokens | One focused question ("Explain CRDTs for collaborative editing") |
| `medium` | ~500 tokens | The current prompt (distributed system architecture) |
| `long` | ~2000 tokens | Extended version with follow-up constraints and examples |

This tells us **at what context length** turbo3 starts breaking. If it's fine at 100 tokens but breaks at 500, that's useful to know.

---

## Number of runs

**3 runs per config** minimum.

Report: median, min, max for each metric. This is enough to show whether the numbers are stable without turning it into a statistics paper.

---

## Parameters (fixed across all runs)

| Parameter | Value | Why |
|-----------|-------|-----|
| `-ngl 99` | All layers on GPU | Removes CPU/GPU split as a variable |
| `-c 8192` | Context window | Large enough to stress KV cache |
| `-n 512` | Tokens to generate | Enough to see quality degradation |
| `--seed 42` | Fixed seed | Deterministic output |
| `--temp 0` | Greedy sampling | Removes sampling randomness |
| `-no-cnv` | Completion mode | No chat template overhead |

---

## What to capture

### Per run

| Data | Source | File |
|------|--------|------|
| KV cache size (K, V, total) | llama.cpp stderr | `{run_id}_output.txt` |
| Prompt eval t/s | `common_perf_print` | `{run_id}_output.txt` |
| Generation t/s | `common_perf_print` | `{run_id}_output.txt` |
| Total time | `common_perf_print` | `{run_id}_output.txt` |
| GPU memory breakdown | `llama_memory_breakdown_print` | `{run_id}_output.txt` |
| Full generated text | stdout | `{run_id}_output.txt` |
| System memory before/after | vm_stat | `{run_id}_memory.txt` |

### Post-run analysis

- **Quality check**: count how many tokens are coherent before degradation (manual inspection or simple heuristic: first non-ASCII-heavy line)
- **Diff between configs**: compare baseline vs turbo3 output to see where they diverge
- **Summary table**: aggregate median/min/max per config per model per prompt length

---

## Execution order

Alternate configs to reduce thermal/system state bias:

```
Run 1: baseline  (short prompt)
Run 2: q8_0      (short prompt)
Run 3: q4_0      (short prompt)
Run 4: turbo3    (short prompt)
Run 5: baseline  (short prompt)  <- repeat for variance
...
```

Wait 5 seconds between runs to let the GPU cool and memory settle.

---

## Script changes needed

The current `run_experiment.sh` needs:

1. Add `q8_0` and `q4_0` to the config options
2. Add a `--prompt` flag to select short/medium/long
3. Default to 3 runs
4. Add a summary output at the end that extracts key metrics into a CSV-friendly format

---

## Expected output

A `RESULTS.md` update with:

```
| Model | Prompt | Config | KV MiB | Gen t/s (median) | Quality |
|-------|--------|--------|--------|-------------------|---------|
| Gemma 3 4B | short  | baseline | 334 | 94.5 | OK |
| Gemma 3 4B | short  | q8_0     | ??? | ???  | ??? |
| Gemma 3 4B | short  | turbo3   | 73  | 58.0 | ??? |
| ...   | ...    | ...    | ...    | ...  | ... |
```

This table is the deliverable — everything else supports it.
