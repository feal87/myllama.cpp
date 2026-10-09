# Expert prediction study

## Dataset

| quantity | value |
| --- | --- |
| recording | `C:/models/expertngrams/ds4.rec` (48.7 MiB) |
| model fingerprint | 8116466990688949793 |
| layers / experts / top-k | 43 / 256 / 6 |
| decode steps | 42183 |
| turns (server requests) | 44 |
| steps per used turn | min 60, median 228, max 11510 |
| largest turn share | 27.3% of all steps |
| distinct decode tokens | 2575 |
| truncated tail | byte 51083697 ignored |

Split: hold out every 5-th turn -> 34913 train steps, 7270 held-out steps.
The marker is one server request (a turn). Turns of one session share context, so
this split measures within-session transfer unless recordings from separate sessions
are supplied; it is not a cross-session or cross-task estimate on its own.
Parameters: orders=1,2,3,4, budgets=1,2,4,6,8,12,16, min_support=1, resident_c=8, holdout_every=5.

## Routing concentration

Static top-8 covers **31.8%** of the routed experts on average across layers (range 7.3% - 51.1%).
Distinct experts seen per layer over the training split: min 249, median 254, max 256 (of 256).

## Temporal locality

Recency: **32.3%** of held-out routed experts were also routed on the immediately previous step.

## Prediction quality

Budgets are per-layer predicted experts. `useful` = predicted, routed and not resident / all routed-not-resident. `waste` = wrong fraction of the prefetched (predicted and not already resident) experts.

### Baselines

| predictor | budget | precision | recall | useful | waste |
| --- | --- | --- | --- | --- | --- |
| oracle | 1 | 1.000 | 0.167 | 0.120 | 0.000 |
| oracle | 2 | 1.000 | 0.333 | 0.257 | 0.000 |
| oracle | 4 | 1.000 | 0.667 | 0.591 | 0.000 |
| oracle | 6 | 1.000 | 1.000 | 1.000 | 0.000 |
| oracle | 8 | 1.000 | 1.000 | 1.000 | 0.000 |
| oracle | 12 | 1.000 | 1.000 | 1.000 | 0.000 |
| oracle | 16 | 1.000 | 1.000 | 1.000 | 0.000 |
| static | 1 | 0.462 | 0.077 | 0.000 | 0.000 |
| static | 2 | 0.398 | 0.133 | 0.000 | 0.000 |
| static | 4 | 0.315 | 0.210 | 0.000 | 0.000 |
| static | 6 | 0.265 | 0.265 | 0.000 | 0.000 |
| static | 8 | 0.230 | 0.306 | 0.000 | 0.000 |
| static | 12 | 0.183 | 0.367 | 0.073 | 0.935 |
| static | 16 | 0.155 | 0.414 | 0.133 | 0.941 |
| recency | 1 | 0.500 | 0.083 | 0.000 | 0.000 |
| recency | 2 | 0.457 | 0.152 | 0.000 | 0.000 |
| recency | 4 | 0.384 | 0.256 | 0.000 | 0.000 |
| recency | 6 | 0.324 | 0.323 | 0.000 | 0.000 |
| recency | 8 | 0.324 | 0.323 | 0.000 | 0.000 |
| recency | 12 | 0.324 | 0.323 | 0.000 | 0.000 |
| recency | 16 | 0.324 | 0.323 | 0.000 | 0.000 |

### Order 1 (held-out key coverage 97.1%)

| predictor | budget | precision | recall | useful | waste |
| --- | --- | --- | --- | --- | --- |
| ngram1 | 1 | 0.781 | 0.126 | 0.079 | 0.207 |
| ngram1 | 2 | 0.722 | 0.234 | 0.153 | 0.275 |
| ngram1 | 4 | 0.619 | 0.401 | 0.293 | 0.403 |
| ngram1 | 6 | 0.533 | 0.518 | 0.412 | 0.508 |
| ngram1 | 8 | 0.457 | 0.579 | 0.470 | 0.604 |
| ngram1 | 12 | 0.357 | 0.656 | 0.551 | 0.720 |
| ngram1 | 16 | 0.295 | 0.704 | 0.608 | 0.783 |
| ngram1+static | 1 | 0.773 | 0.129 | 0.079 | 0.207 |
| ngram1+static | 2 | 0.713 | 0.238 | 0.153 | 0.275 |
| ngram1+static | 4 | 0.610 | 0.407 | 0.293 | 0.403 |
| ngram1+static | 6 | 0.526 | 0.526 | 0.412 | 0.508 |
| ngram1+static | 8 | 0.451 | 0.588 | 0.470 | 0.604 |
| ngram1+static | 12 | 0.351 | 0.666 | 0.553 | 0.724 |
| ngram1+static | 16 | 0.291 | 0.716 | 0.611 | 0.786 |

### Order 2 (held-out key coverage 79.5%)

| predictor | budget | precision | recall | useful | waste |
| --- | --- | --- | --- | --- | --- |
| ngram2 | 1 | 0.783 | 0.104 | 0.066 | 0.257 |
| ngram2 | 2 | 0.738 | 0.196 | 0.133 | 0.308 |
| ngram2 | 4 | 0.649 | 0.344 | 0.264 | 0.410 |
| ngram2 | 6 | 0.567 | 0.451 | 0.375 | 0.503 |
| ngram2 | 8 | 0.491 | 0.498 | 0.423 | 0.591 |
| ngram2 | 12 | 0.393 | 0.552 | 0.483 | 0.696 |
| ngram2 | 16 | 0.334 | 0.582 | 0.519 | 0.752 |
| ngram2+static | 1 | 0.735 | 0.122 | 0.066 | 0.257 |
| ngram2+static | 2 | 0.682 | 0.227 | 0.133 | 0.308 |
| ngram2+static | 4 | 0.589 | 0.393 | 0.264 | 0.410 |
| ngram2+static | 6 | 0.512 | 0.512 | 0.375 | 0.503 |
| ngram2+static | 8 | 0.441 | 0.567 | 0.423 | 0.591 |
| ngram2+static | 12 | 0.349 | 0.634 | 0.498 | 0.726 |
| ngram2+static | 16 | 0.294 | 0.673 | 0.547 | 0.786 |

### Order 3 (held-out key coverage 62.1%)

| predictor | budget | precision | recall | useful | waste |
| --- | --- | --- | --- | --- | --- |
| ngram3 | 1 | 0.754 | 0.078 | 0.051 | 0.312 |
| ngram3 | 2 | 0.712 | 0.147 | 0.104 | 0.355 |
| ngram3 | 4 | 0.631 | 0.261 | 0.208 | 0.443 |
| ngram3 | 6 | 0.557 | 0.346 | 0.298 | 0.522 |
| ngram3 | 8 | 0.486 | 0.380 | 0.334 | 0.601 |
| ngram3 | 12 | 0.396 | 0.419 | 0.379 | 0.693 |
| ngram3 | 16 | 0.343 | 0.440 | 0.404 | 0.744 |
| ngram3+static | 1 | 0.675 | 0.112 | 0.051 | 0.312 |
| ngram3+static | 2 | 0.618 | 0.206 | 0.104 | 0.355 |
| ngram3+static | 4 | 0.529 | 0.352 | 0.208 | 0.443 |
| ngram3+static | 6 | 0.460 | 0.460 | 0.298 | 0.522 |
| ngram3+static | 8 | 0.396 | 0.511 | 0.334 | 0.601 |
| ngram3+static | 12 | 0.315 | 0.573 | 0.407 | 0.754 |
| ngram3+static | 16 | 0.267 | 0.612 | 0.456 | 0.812 |

### Order 4 (held-out key coverage 50.6%)

| predictor | budget | precision | recall | useful | waste |
| --- | --- | --- | --- | --- | --- |
| ngram4 | 1 | 0.725 | 0.061 | 0.043 | 0.348 |
| ngram4 | 2 | 0.686 | 0.116 | 0.086 | 0.389 |
| ngram4 | 4 | 0.609 | 0.206 | 0.173 | 0.469 |
| ngram4 | 6 | 0.541 | 0.274 | 0.247 | 0.540 |
| ngram4 | 8 | 0.474 | 0.301 | 0.277 | 0.612 |
| ngram4 | 12 | 0.389 | 0.332 | 0.312 | 0.696 |
| ngram4 | 16 | 0.341 | 0.348 | 0.332 | 0.741 |
| ngram4+static | 1 | 0.634 | 0.106 | 0.043 | 0.348 |
| ngram4+static | 2 | 0.575 | 0.192 | 0.086 | 0.389 |
| ngram4+static | 4 | 0.486 | 0.324 | 0.173 | 0.469 |
| ngram4+static | 6 | 0.422 | 0.422 | 0.247 | 0.540 |
| ngram4+static | 8 | 0.364 | 0.471 | 0.277 | 0.612 |
| ngram4+static | 12 | 0.290 | 0.533 | 0.349 | 0.778 |
| ngram4+static | 16 | 0.245 | 0.572 | 0.399 | 0.833 |

### Backoff (largest matching order, else static)

| predictor | budget | precision | recall | useful | waste |
| --- | --- | --- | --- | --- | --- |
| backoff | 1 | 0.744 | 0.124 | 0.084 | 0.308 |
| backoff | 2 | 0.707 | 0.236 | 0.168 | 0.348 |
| backoff | 4 | 0.629 | 0.420 | 0.330 | 0.434 |
| backoff | 6 | 0.557 | 0.557 | 0.468 | 0.514 |
| backoff | 8 | 0.487 | 0.609 | 0.520 | 0.593 |
| backoff | 12 | 0.398 | 0.667 | 0.581 | 0.692 |
| backoff | 16 | 0.346 | 0.697 | 0.615 | 0.745 |

## Per-layer breakdown (backoff, budget 16)

| layer | precision | recall | useful |
| --- | --- | --- | --- |
| 0 | 0.928 | 0.973 | 0.971 |
| 1 | 0.928 | 0.973 | 0.971 |
| 2 | 0.928 | 0.973 | 0.970 |
| 3 | 0.434 | 0.811 | 0.783 |
| 4 | 0.436 | 0.831 | 0.790 |
| 5 | 0.383 | 0.762 | 0.715 |
| 6 | 0.394 | 0.780 | 0.724 |
| 7 | 0.368 | 0.745 | 0.682 |
| 8 | 0.388 | 0.769 | 0.696 |
| 9 | 0.377 | 0.752 | 0.690 |
| 10 | 0.389 | 0.766 | 0.710 |
| 11 | 0.320 | 0.665 | 0.582 |
| 12 | 0.310 | 0.650 | 0.557 |
| 13 | 0.302 | 0.639 | 0.491 |
| 14 | 0.310 | 0.653 | 0.536 |
| 15 | 0.296 | 0.630 | 0.518 |
| 16 | 0.274 | 0.590 | 0.480 |
| 17 | 0.278 | 0.597 | 0.475 |
| 18 | 0.281 | 0.605 | 0.522 |
| 19 | 0.444 | 0.837 | 0.815 |
| 20 | 0.302 | 0.642 | 0.566 |
| 21 | 0.304 | 0.647 | 0.550 |
| 22 | 0.290 | 0.618 | 0.539 |
| 23 | 0.296 | 0.634 | 0.465 |
| 24 | 0.297 | 0.637 | 0.386 |
| 25 | 0.272 | 0.592 | 0.363 |
| 26 | 0.305 | 0.656 | 0.374 |
| 27 | 0.288 | 0.624 | 0.443 |
| 28 | 0.296 | 0.636 | 0.420 |
| 29 | 0.301 | 0.644 | 0.475 |
| 30 | 0.283 | 0.609 | 0.443 |
| 31 | 0.289 | 0.619 | 0.493 |
| 32 | 0.284 | 0.609 | 0.428 |
| 33 | 0.304 | 0.646 | 0.478 |
| 34 | 0.290 | 0.624 | 0.463 |
| 35 | 0.335 | 0.696 | 0.603 |
| 36 | 0.279 | 0.604 | 0.430 |
| 37 | 0.314 | 0.664 | 0.538 |
| 38 | 0.306 | 0.652 | 0.492 |
| 39 | 0.351 | 0.724 | 0.632 |
| 40 | 0.332 | 0.700 | 0.556 |
| 41 | 0.346 | 0.724 | 0.655 |
| 42 | 0.367 | 0.749 | 0.688 |

## Per-turn variability (backoff, budget 16)

| turn | steps | precision | recall | useful |
| --- | --- | --- | --- | --- |
| 0 | 118 | 0.264 | 0.637 | 0.560 |
| 5 | 267 | 0.362 | 0.710 | 0.648 |
| 10 | 650 | 0.344 | 0.697 | 0.626 |
| 15 | 3694 | 0.377 | 0.733 | 0.651 |
| 20 | 1583 | 0.301 | 0.645 | 0.573 |
| 25 | 471 | 0.309 | 0.628 | 0.541 |
| 30 | 111 | 0.332 | 0.708 | 0.643 |
| 35 | 187 | 0.340 | 0.685 | 0.606 |
| 40 | 189 | 0.298 | 0.621 | 0.539 |

## Prefetch benefit vs cache residency

The previous step is always resident. As the static set grows the demand misses shrink and the prefetch headroom with them. `reads useful` is the fraction of the real prefetches that hit a routed expert.

| static experts/layer | demand misses/step | backoff useful (budget 6) | reads/step | reads useful |
| --- | --- | --- | --- | --- |
| 2 | 163.8 | 49.1% | 170.9 | 47.1% |
| 4 | 154.5 | 47.9% | 155.9 | 47.4% |
| 8 | 140.5 | 46.8% | 135.2 | 48.6% |
| 16 | 121.8 | 45.9% | 113.1 | 49.5% |
| 32 | 97.5 | 44.9% | 87.2 | 50.2% |
| 64 | 64.5 | 42.9% | 54.6 | 50.6% |
| 128 | 25.0 | 41.4% | 19.8 | 52.3% |

## Prefetch volume (backoff)

`reads/step` is the prefetched experts not already resident (the real reads), `covered` is the fraction of demand misses the prediction reaches, `reads useful` the fraction of reads that hit a routed expert.

| budget | predicted/step | reads/step | demand misses/step | covered | reads useful |
| --- | --- | --- | --- | --- | --- |
| 1 | 43.0 | 17.1 | 140.5 | 8.4% | 69.2% |
| 2 | 86.0 | 36.2 | 140.5 | 16.8% | 65.2% |
| 4 | 172.0 | 82.0 | 140.5 | 33.0% | 56.6% |
| 6 | 258.0 | 135.2 | 140.5 | 46.8% | 48.6% |
| 8 | 322.9 | 179.4 | 140.5 | 52.0% | 40.7% |
| 12 | 432.7 | 265.1 | 140.5 | 58.1% | 30.8% |
| 16 | 520.1 | 338.5 | 140.5 | 61.5% | 25.5% |

