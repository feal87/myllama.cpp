# Disk-stage sparse/full prefill threshold

The disk decode cache fills a prefill ubatch in one of two ways:

- **sparse** (`fill_selected`): read only the experts the ubatch routes, and
  copy the ones already resident in the RAM decode cache.
- **full** (`fill`): stream the whole layer slab, resident experts included.

Small ubatches route few experts, so sparse wins. Large ones touch almost every
expert, so the sequential slab read wins. The cut is decided per ubatch, on the
ubatch token count (`n_ubatch`), not on the prompt length.

Engine flag: `--disk-stage-sparse-max N`. A multi-token ubatch smaller than `N`
reads sparse, `N` or more streams the slab. `0` always streams. Without the flag
the engine falls back to `LLAMA_DISK_STAGE_SPARSE_MAX`, then to `32`.

The switcher exposes it per profile as `diskStageSparseMax` (`--disk-stage-sparse-max`).

## Why the crossover is per model

Sparse only reads the experts that are **not** already in the RAM disk decode
cache, so its win is set by how much of the routed set is resident. The larger
the cache coverage (`resident slots per layer / n_expert`), the further sparse
wins. The full path is one sequential read per ubatch, so it is flat.

Measured on this host (RTX 4060 Ti 8 GiB, 68 GiB RAM, NVMe), warm cache, real
text prompts:

| model | ubatch | RAM residents | coverage | full per ubatch | crossover |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 3072 | 10909/17616 (48/48) | ~367/512 = 72% | ~8.1s | > 3072 (always sparse) |
| GLM-5.3-Flash | 3072 | 3360/3360 (42/42) | ~80/288 = 28% | ~19s | ~120 |
| Mimo26 | 2048 | 3948/3948 (47/47) | ~84/256 = 33% | ~15.2s | ~125 |
| DeepSeek-V4-Flash | 2304 | 3999/3999 (43/43) | ~93/256 = 36% | ~13.6s | ~190 |

Current profile values: `code` 4096, `creative` 8192, `glm53` 120, `mimo`/`mimo-code`
125, `dsv4`/`dsv4_code` 190.

## Procedure for a new model

1. Add/keep the profile with `diskStageSparseMax` unset (engine default 32, i.e.
   full for every real ubatch).
2. Launch it through the switcher and **warm with a real open-ended prompt**:
   `python warm.py --base http://127.0.0.1:1235 --tokens 500`.
   Never warm with random tokens: the model collapses into a repetition loop,
   routes a handful of experts, and the cache never fills. Verify the RAM tier
   report shows `layers N/N` and a large resident count before measuring.
3. Measure the **full** curve: set `diskStageSparseMax` to `0`, restart, warm,
   then `python sweep.py 64 128 200 300 430 700 1200 2000 --files <corpus...>`.
   It is flat; a couple of points are enough.
4. Measure the **sparse** curve: set `diskStageSparseMax` above the ubatch size
   (e.g. `100000`), restart, warm, sweep again. Extend `n` past `n_ubatch` to see
   the staircase: full costs `ceil(n / n_ubatch) * full_per_ubatch`, sparse does
   not.
5. Crossover = the `n` where sparse crosses the flat full time. Set
   `diskStageSparseMax` to that value (strict `<`, so it must be above the value
   you want sparse).

`sweep.py` builds the prompt pool from real text files (for example the repo
`README.md` plus `docs/*.md`), tokenizes it once, and sends growing prefixes with
`cache_prompt: false` and `n_predict: 1`, printing `prompt_ms`.

## Notes

- Warm-up must use a real prompt. A random-token warm makes the model emit a
  degenerate completion, which concentrates routing and pins the VRAM MoE layout
  onto a handful of experts; the RAM cache then never fills and every number is
  meaningless.
- Split-hot and the drop/substitute/L2 options are single-token decode only
  (`dc == nullptr` for `n_tokens > 1` in `llama-graph.cpp`). They do not affect
  prefill and do not need to be toggled for this sweep.
- The warm decode cache content still matters, because `fill_run` copies
  resident experts in both paths.
