# Findings: ds4.rec (DeepSeek V4 Flash, one coding session)

Source: `C:\models\expertngrams\ds4.rec`, recorded by the `dsv4_code` profile.
Auto-generated numbers: `ds4-report.md` / `ds4-report.json` (regenerate with the
command in `README.md`; the run is deterministic).

## What the recording actually is

This is **one coding session**, not 44 independent prompts: a task to make a page
responsive. The 44 markers are the 44 server requests (turns) of that session -
user message and tool results in, generated code out - so consecutive turns share
context and vocabulary. 42,183 of the ~100K tokens were decode; the rest were
prefill. The session is prefill-dominated.

This is exactly the shape a coding-agent / IDE workflow produces with prompt
caching, so it is the right workload to study. The consequence for the analysis
is that a turn split measures *within-session* transfer, which is the realistic
deployment case for a long session, but not cross-task generalisation.

## The numbers are stable to the split

The result barely moves when the split changes, which is a good sign that it is a
property of the routing, not an artifact of interpolation:

| split | useful (backoff, M=6) |
| --- | --- |
| every 5th turn held out | 0.468 |
| temporal: train turns 0-34, test 35-43 | 0.452 |
| train later turns, test the first 4 turns | 0.449 |

## Headline

At a budget of 6 predicted experts per layer, the best predictor (largest matching
n-gram order, else static top-6) covers 46.8% of the experts that would otherwise
be demand reads, but only 48.6% of the prefetch reads are correct: it reads ~135
experts per step to hide ~66 demand reads, and misses the other ~75.

## Per-layer precision is a real and stable anchor

This is the most actionable finding and it supports the per-layer idea:

- Per-layer usefulness is reproducible: the correlation between two disjoint
  halves of the session is **0.993**, and between the held-out turns and the mean
  of the halves **0.995**. A fraction of the session is enough to estimate it.
- A selector built only on the training turns (even turns) and applied to the
  held-out turns matches the oracle selector (ranked on the held-out turns)
  almost exactly. So selecting layers from offline stats is viable.
- Demanding fewer layers raises prefetch efficiency sharply. At budget 6:

  | layers predicted | reads/step | demand misses covered | reads useful |
  | --- | --- | --- | --- |
  | best 3 (0-2) | 15.6 | 11.1% | **100.0%** |
  | best 5 | 23.4 | 14.8% | 89.1% |
  | best 8 | 33.9 | 19.1% | 79.1% |
  | best 16 | 60.6 | 28.8% | 66.8% |
  | best 22 | 79.9 | 34.2% | 60.1% |
  | all 43 | 135.2 | 46.8% | 48.6% |

- The best layers are structural: **0, 1, 2** are near-perfect (rounding to 1.0
  reads useful), then 19, 3, 4, 6, 10, 8. The worst are 25, 26, 24, 23 at
  ~0.20-0.24. Early layers are strongly token-determined (low static coverage
  but near-deterministic routing), so that part should transfer across tasks.
  The middle/late ordering is more likely to be session-specific and must be
  re-checked on other sessions.

The correct anchor is not cumulative precision but the layer's *marginal*
efficiency (`reads useful`) combined with its volume (`demand misses/step`),
because a layer with high precision and few misses has little to contribute.
Under that lens, layers 0-2 and 19 are the clear targets, and the bulk of the
remaining helpful reads comes from the many mid layers at moderate efficiency.

## Interpretation

The idea is directionally sound and, importantly, the layer signal is stable
enough to act on offline. But the predictor roughly doubles expert reads to
remove about half of the decode stalls, on an unusually repetitive workload
(2,575 distinct tokens). Whether that is a win depends on the disk stage: if
demand reads are latency-critical and the extra reads can run in parallel it can
pay off; if the disk stage already overlaps reads with compute, the extra reads
are pure bandwidth.

## Next steps

1. **Record more sessions** (different tasks and projects) and add a cross-file
   split to `study.py` so train is one session and eval another. This answers
   whether the middle-layer selection and the ~47% coverage transfer. Use one
   recording per session; the format currently marks run starts and turns with
   the same record type, so a distinct session record is worth adding.
2. **Value the hit in time, not counts.** Instrument the disk stage to report,
   per layer, the demand-read stall time saved and the extra bytes read, then
   rerun. That turns `reads useful` into a throughput estimate.
3. **Prototype the selective prefetch**: a per-layer allowlist plus a modest
   budget (start with layers 0-2, then add by measured efficiency), guarded by a
   global per-step read cap, and A/B decode throughput.
4. **Consider an online selector** (decayed per-layer hit rate) so the policy
   adapts within a session without a pre-built database.
5. If cross-session transfer is weak, only the structural early layers survive;
   a learned predictor over context embeddings would be the follow-up.
