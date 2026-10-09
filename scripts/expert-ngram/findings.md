# Findings: coding-workflow expert prediction (two DS4 sessions)

Recordings: `ds4_1.rec` (42,183 decode steps, 44 turns) and `ds4.rec` (72,231
decode steps, 146 turns), both DeepSeek V4 Flash, both coding sessions from the
`dsv4_code` profile. 114,414 decode steps, 190 turns, 3,973 distinct tokens.
Auto-generated numbers: `combined-report.md` / `combined-report.json`.

## Headline

The single-session result stands when the second session is added, and it
survives a true cross-session test:

| evaluation | precision | recall | useful | reads useful |
| --- | --- | --- | --- | --- |
| turn split, both sessions (within-session) | 0.532 | 0.532 | 0.456 | 48.1% |
| cross-session: train ds4, test ds4_1 | 0.403 | 0.403 | 0.327 | 39.4% |
| cross-session: train ds4_1, test ds4 | 0.323 | 0.323 | 0.235 | 38.0% |
| cross-session mean | 0.352 | 0.352 | 0.266 | 38.6% |

Budget 6 (top-k), background predictor = largest matching n-gram order 1..4,
else static top-6. `reads useful` is the fraction of real prefetches that hit a
routed expert.

So a profile built on one coding session predicts another coding session at
**~35% precision and ~39% prefetch efficiency**, down from ~53% / ~48% within the
same session. The transfer is real, not just vocabulary echo: order-1 key
coverage on the held-out session is 0.71-0.81, but precision is measured against
the experts actually routed, and the static baseline stays at useful 0.

## The per-layer anchor transfers almost perfectly

Correlation of per-layer usefulness between the turn-split ranking and the
cross-session ranking: **0.982** and **0.978**. The layers that are predictable in
one coding session are predictable in the other. This is the strongest result in
the study:

- Layers 0, 1, 2 are near-deterministic (token-determined routing right after the
  embedding) and stay that way across sessions and tasks.
- The middle-layer ordering is also stable across these two sessions, but both
  are the same workflow and model, so re-check on a prose session before trusting
  it.
- A selector built from training turns matched the oracle selector in the earlier
  single-session study, and the ranking now transfers, so a per-layer allowlist
  can be baked into a profile.

Selective prefetch at budget 6 (single-session study): best 3 layers -> 100%
reads useful and 11% coverage; best 8 -> 79% and 19%; all 43 -> 48.6% and 47%.
There is a tunable coverage/efficiency frontier, and the per-layer choice is
stable enough to pick offline.

## Split sensitivity

The turn-split number (0.456) is close to the single-session values (0.468 for
session 1; 0.452 temporal) and to the cross-session numbers, so the effect is a
property of the routing rather than of one split.

## Proposed approach: per-workflow profiles

The data supports a profile-per-workflow design. Concretely:

### Profile contents

Build one profile per workflow (coding, prose, ...) from several sessions of that
workflow. A profile holds:

- a model/tokenizer fingerprint (reject on mismatch),
- the n-gram orders and the per-order top-M expert sets (backoff),
- the per-layer allowlist (which layers are worth prefetching),
- a per-layer budget, so predictable layers get more than noisy ones.

### On-disk format (fast, no search)

- Header: magic, version, model fingerprint, `n_layer`, `n_expert`, `n_expert_used`,
  orders, top-M, enabled-layer mask, table size.
- One open-addressed hash table, power-of-two size, load factor below 0.5,
  linear probing, storing the full 64-bit key (0 = empty). Lookup is one hash and
  a probe or two.
- Per entry: a per-layer expert bitmask for the enabled layers. For 256 experts a
  mask is 32 bytes, so 12 enabled layers is 384 bytes/entry; at 200k keys and 2x
  load, about 150 MiB. Fixed-size records keep it cache-friendly.
- `mmap` the file and read it in place: no parse, no per-key allocation at
  startup.

### Runtime lookup and layer-order loading

- Keep one rolling hash per order, updated with the current token in O(orders)
  time (polynomial sliding-window hash). Key = `hash(order, window)`, so orders
  never alias.
- At each single-token decode step, before the graph runs, try orders from the
  largest down; the first hit wins (backoff). One lookup, at most `orders`
  probes.
- From the entry, iterate layers 0..L-1 in execution order and issue async reads
  for the enabled layers only, in that order, so the experts closest to being
  executed are read first and the rest overlap earlier compute. Cap the total
  reads per step to bound the I/O queue.
- Skip experts already resident (static set + previous step), and skip layers not
  in the allowlist, so the wasted reads are bounded.

### Why this shape

- The per-layer ranking transfers (0.98), so the allowlist is safe to bake in.
- The bitmask payload makes layer-order iteration trivial and avoids per-entry
  variable sizes.
- The hash table plus mmap avoids the "search the database" overhead entirely:
  it is one arithmetic hash and one memory probe per step.
- One profile per workflow lets a coding profile include only coding layers and
  key distributions, and a prose profile its own, instead of averaging them.

## Open questions before wiring it in

1. **Does it pay off in time?** The offline numbers say ~39% of cross-session
   prefetch reads hit, at roughly 2x the expert reads. That is only a win if the
   demand reads are latency-critical and the extra reads overlap. Instrument the
   disk stage to report per-layer stall time saved versus extra bytes, then
   rerun.
2. **Prose / other workflows.** Record a non-coding session and repeat. If the
   per-layer allowlist and coverage differ much, profiles per workflow are
   justified; if they are similar, one combined profile may do.
3. **Minimum profile size.** How many sessions/tokens are needed before the
   cross-session numbers stabilize. The per-layer ranking looks estimable from a
   fraction of one session; the key coverage needs more.

## Next steps

1. Export profiles: add a `--export` to `study.py` that writes the binary profile
   above from the recordings (with the allowlist and per-layer budgets).
2. Record a prose session and rerun both the cross-session and the per-layer
   stability analysis.
3. Prototype the runtime: `--expert-ngram-profile FILE`, plan-then-issue in layer
   order, A/B decode throughput and per-layer hit rate on the real disk stage.
4. Then decide whether to keep exact n-grams or move to a small learned predictor
   if cross-workflow transfer turns out weak.
