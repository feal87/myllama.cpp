# Expert prediction study (n-gram)

Experimental: given the last N decode tokens, can we predict which MoE experts
the next graph will route, early enough to prefetch them? This directory only
gathers data and evaluates offline; it does not change inference.

Two pieces:

- `--expert-ngram-record FILE` (engine flag) appends a raw binary recording of
  every single-token decode step: the token the graph consumed and, per MoE
  layer, the experts it routed.
- `study.py` builds an n-gram -> expert model on a training split of the prompts
  and scores prediction quality on a held-out split, against the baselines that
  matter. `analyze.py` is a dependency-free (stdlib + sqlite) quick look for
  small files; `study.py` is the deep, vectorized one (requires numpy).

## Gather

Run `-np 1` so multiple server slots do not interleave their token streams:

```sh
llama-server -m model.gguf --load-mode dio --expert-ngram-record run.rec -np 1
```

Or via the model switcher: set `expertNgramRecord` on the profile (path relative
to `modelRoot`) and `parallel: 1`.

The file is opened in append mode and is safe to feed across several sessions;
each run writes a prompt-boundary marker first, and within a run every prompt
switch writes another, so the analysis never builds an n-gram across two
unrelated prompts. The header carries a model/tokenizer fingerprint: appending to
a file made by another model disables recording instead of corrupting it. A file
whose engine was killed mid-write can end with a truncated record; the parsers
ignore the tail and report the byte offset.

## Deep study (recommended)

Pass one recording per session. With more than one file the report adds a
leave-one-session-out transfer table, which is the cross-session test a
per-workflow profile has to survive.

```sh
python scripts/expert-ngram/study.py session1.rec session2.rec \
    --orders 1 2 3 4 \
    --top-m 1 2 4 6 8 12 16 \
    --min-support 1 \
    --holdout-every 5 \
    --resident-c 8 \
    --report report.md --json report.json
```

The run is deterministic: the split is `holdout-every K` on the turn index and
leave-one-session-out on the files, with no sampling, so the same recordings
always yield the same report.

### Method

- **Split**: two splits are reported. `turns` holds out every K-th server
  request across all files. `cross` treats each file as a session and does
  leave-one-session-out. Turns of one session share context, so the turn split
  is within-session transfer; the cross split is the per-workflow transfer.
- **Key**: the exact last-N decode tokens ending at the token the step consumed.
  Keys never span a turn or session boundary.
- **Model**: per (key, layer), the top-M routed experts by training frequency,
  pruned by `min-support`.
- **Predictors**: `static` (per-layer top-M over all training routes), `recency`
  (the previous step's experts), `oracle`, `ngramN`, `ngramN+static`, and
  `backoff` (largest matching order, else static).
- **Resident model**: an expert is assumed already available when it is in the
  static top-`resident-c` per layer OR was routed on the immediately previous
  step. Those are the experts a real cache would already serve, so predicting
  them is worthless. `oracle` and `static` prove the modeling: `oracle` has
  `useful = 1`, `static` has `useful = 0` while it only predicts residents.

### Metrics

- `precision` = predicted experts that were routed / predicted.
- `recall` = predicted experts that were routed / routed.
- `useful` = predicted AND routed AND not-resident / all routed-not-resident.
  This is the fraction of demand misses a successful prefetch could have hidden.
- `waste` = wrong predicted experts among the prefetched (predicted and
  not-resident) / all prefetched. `1 - waste` is the fraction of real prefetch
  reads that paid off.
- A budget above top-k saturates (the extra slots cannot exist).

### Reading the result

The whole answer depends on how much the cache already holds, so the
"Prefetch benefit vs cache residency" table sweeps `resident-c`. As the cache
grows, demand misses shrink and the prefetch headroom shrinks with them. If
`static` and `recency` already reach high `useful` for your real cache size, the
predictor adds little. If `ngram` recall only rises with high `waste`, prefetch
would read mostly wrong experts.

## Export a profile

```sh
python scripts/expert-ngram/export_profile.py coding.prof session1.rec session2.rec \
    --orders 1 2 3 4 --top-m 6 --layer-threshold 0.40 --verify
```

The exporter builds the table from every turn of the inputs, and chooses the
per-layer allowlist from leave-one-session-out folds (a turn-parity split when
there is one file). `--sweep` prints the per-layer efficiency and the realized
coverage/efficiency for several thresholds without writing a file.

### Choosing the layer threshold

The threshold is the minimum per-read efficiency (prefetch reads that hit /
reads) for a layer to be enabled. It is an I/O economics choice, not a
correctness bar, so it is a knob. `--sweep` on the two DS4 coding sessions gave:

| threshold | layers | reads/step | demand misses covered | reads useful |
| --- | --- | --- | --- | --- |
| 0.25 | 25 | 72.7 | 21.4% | 47.5% |
| 0.30 | 17 | 52.2 | 17.9% | 55.2% |
| 0.35 | 16 | 49.4 | 17.4% | 56.6% |
| 0.40 | 10 | 32.9 | 13.5% | 66.1% |
| 0.45 | 6 | 21.6 | 10.6% | 79.2% |
| 0.50 | 5 | 18.3 | 9.6% | 84.5% |
| 0.60 | 3 | 12.3 | 7.6% | 100.0% |

Start at 0.40-0.45 (the knee): roughly two thirds to four fifths of the reads pay
off. Drop toward 0.30 only if a hidden stall is worth much more than a read
(NVMe, parallel reads); raise toward 0.50 on a bandwidth-bound disk. The layer
ranking is stable across sessions, so the choice transfers.

## Profile format (`LENGPROF` v1, little-endian)

Header, 68 bytes:

| offset | size | field |
| --- | --- | --- |
| 0 | 8 | magic `LENGPROF` |
| 8 | 4 | version (1) |
| 12 | 4 | n_layer |
| 16 | 4 | n_expert |
| 20 | 4 | n_expert_used |
| 24 | 4 | max_order |
| 28 | 4 | top_m |
| 32 | 4 | table_size (power of two) |
| 36 | 4 | n_enabled |
| 40 | 4 | layer_mask_bytes = ceil(n_layer/8) |
| 44 | 8 | model fingerprint |
| 52 | 8 | tokenizer fingerprint |
| 60 | 8 | n_entries |

Then, in order:

- `layer_mask[layer_mask_bytes]`: bit `il` set = layer enabled.
- `budgets[n_layer]` as u32: per-layer cap on prefetched experts (0 = disabled).
- `slots[table_size]`, each `u64 key` followed by `n_enabled * top_m` u16 expert
  ids (`0xFFFF` = empty). The payload is in ascending enabled-layer order, so the
  runtime iterates layers in execution order. Empty slot = key 0.

The key hash must match the exporter exactly:

```
h = 0xcbf29ce484222325
for v in window (oldest first, `order` values):
    h ^= (u32) v
    h *= 0x100000001b3            # mod 2^64
h ^= (order * 0x9e3779b97f4a7c15) # mod 2^64
h *= 0x100000001b3
h ^= h >> 33
h *= 0xff51afd7ed558ccd
h ^= h >> 33
if h == 0: h = 1
```

Lookup: for `order` from `max_order` down to 1, hash the last `order` tokens and
probe linearly from `h & (table_size - 1)`; the first order with a hit wins
(backoff). Then iterate layers 0..n_layer-1 and, for each enabled layer, read its
`top_m` ids from the payload (ordinal = popcount of the mask below that layer),
skip `0xFFFF`, cap at `budgets[il]`, and issue the reads in that order.

`export_profile.py --verify` re-reads the file and round-trips sampled training
keys against the source model.

## Quick look (analyze.py)

```sh
python scripts/expert-ngram/analyze.py run.rec --orders 1 2 3 --top-m 1 2 4 8 --holdout-every 5
```

Same split and metrics, stdlib only, no per-layer or residency output. Use it for
a sanity check; use `study.py` for a report.

## Recording format (`LENGREC1`, little-endian)

Header, written once (40 bytes):

| field | type | meaning |
|---|---|---|
| magic | char[8] | `LENGREC1` |
| version | u32 | 1 |
| n_layer | u32 | transformer layers |
| n_expert | u32 | experts per MoE layer |
| n_expert_used | u32 | top-k of the router |
| model_fp | u64 | arch + layer/expert/embd fingerprint |
| tokenizer_fp | u64 | tokenizer fingerprint |

Records, appended one after another:

- `u8 type = 1` (prompt begin): no payload.
- `u8 type = 2` (decode step):
  - `i32 token`: the token the graph consumed this step;
  - `u16 n_layer_obs`: number of observed layers;
  - for each: `u16 layer`, `u16 k`, `i32 ids[k]` (the routed experts).

The two 16-bit fields bound a recording to < 65536 layers/experts and a top-k
below that, which no current model approaches.

## Limits

- Exact token n-grams do not generalise: an unseen key predicts nothing, and a
  short key collides. The order sweep finds the useful compromise.
- Routing partly depends on long-range semantic context an n-gram key cannot
  see, which caps accuracy regardless of order.
- The analysis treats every turn as restarting from the static resident set; it
  does not model what a real session carries across turns beyond the previous
  step.
- A single recording is one session, so the split measures within-session
  transfer. Cross-session and cross-task generalisation need more recordings.
- The resident model is a static top-C plus the previous step, not the real
  dynamic promotion/eviction policy. The residency sweep is a proxy.
- The model is built in memory and dropped at the end; it is not yet exported as
  a runtime database file.

## Runtime prefetch

`--expert-ngram-profile FILE` makes a single-token decode step read ahead:

```
llama-server -m model.gguf --load-mode dio --pin-hot-experts 400 \
    --expert-ngram-profile expertngrams/coding.prof
```

The step hashes the recent input tokens against the table (largest order first,
first hit wins), then the disk stage reads the non-resident predicted experts
into the L2 pool from a dedicated worker on its own completion port, so the
read overlaps the graph instead of queueing behind a layer's demand read. The
per-step read count is capped by `--expert-ngram-prefetch-max` (default 64).

The hot-expert report gains a `predict` line per interval: steps matched,
experts predicted, and the share the step actually routed (precision), plus the
per-layer hit rate. That is the number to watch in the A/B:

```
experts stats:
  predict  : 44/44 steps matched | 264 predicted, 121 routed (45.8% precision)
  predict/L: L0=5/6 L1=5/6 L2=4/6 ...
```

Tune the profile's `--layer-threshold` together with the runtime: with a fast
NVMe a lower threshold (more layers, more reads) can win, on a slower drive a
higher one. The exporter's `--sweep` prints the coverage/efficiency frontier to
pick a starting point.
