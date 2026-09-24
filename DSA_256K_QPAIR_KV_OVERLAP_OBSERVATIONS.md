# Adjacent-Query KV-Index Overlap in 256K DSA Training

## TL;DR

- A 10-step, 256K-token DSA training run measured the exact KV-index-set
  overlap of non-overlapping adjacent query pairs: `(q0, q1), (q2, q3), ...`.
- The all-layer mean overlap rose from 13.18% on the first forward pass to
  50.51% after one optimizer update and reached roughly 72% to 75% from step 3
  onward.
- At step 9, 92.81% of query pairs shared at least half of their valid selected
  KV indices, and the median pair overlap was 77.29%.
- The signal is strongly layer-dependent. Layer 1 remained near 5% to 7%, while
  23 of 24 layers exceeded 50% mean overlap at step 9.
- The result supports an adaptive q-cluster path, but not a model-wide fixed
  shared-index count. A per-layer or per-batch gate is needed to preserve a
  cheap fallback for low-overlap inputs.

This was a short experiment from random initialization, not a converged-model
measurement. It establishes the measurement method and reveals a strong early
q-cluster signal; production thresholds still need calibration on the target
checkpoint and data distribution.

## 1. Question

Q-cluster pairs adjacent query rows and attempts to process their common sparse
KV selections together. Its useful-work reduction depends on how many indices
the two top-k sets actually share.

For adjacent queries with selected KV-index sets `A` and `B`, the primary
measurement was:

```math
\mathrm{overlap}(A,B)
=
\frac{|A\cap B|}{\min(|A|,|B|)}.
```

The denominator uses the valid top-k length of each row. It therefore remains
well-defined in the causal prefix, where fewer than 2,048 KV rows are available.
The experiment also recorded Jaccard overlap:

```math
\mathrm{Jaccard}(A,B)
=
\frac{|A\cap B|}{|A\cup B|}.
```

The pairing is deliberately non-overlapping: `(q0, q1), (q2, q3), ...`. This
matches a two-query q-cluster execution model and avoids counting one query in
two clusters. Context-parallel or variable-length discontinuities are excluded
rather than being treated as adjacent global queries.

## 2. Experiment setup

| Parameter | Value |
| --- | --- |
| Hardware | 8 NVIDIA B300 GPUs |
| Model recipe | Ling Tiny v3-style DSA |
| Transformer layers | 24 |
| Hidden size | 1,536 |
| DSA indexer heads | 16 |
| Absorbed query/key width | 576 |
| Sequence length | 262,144 |
| Sparse top-k | 2,048 |
| Context parallelism | 8 |
| Training steps | 10 |
| Global batch size | 1 |
| Precision | BF16, with FP8 DSA indexer path |
| Data | 0.5 WikiText-2 + 0.5 Stanford Alpaca |
| Initialization | Random |

Step 0 is the first forward pass before the first optimizer update. Step 1 is
measured after one update, and so on. The run consumed approximately 2.62
million training tokens.

## 3. Measurement implementation

The instrumentation observes the canonical top-k indices after they have been
sorted by KV index and before sparse attention consumes them. For each layer and
rank it:

1. forms adjacent query pairs;
2. computes exact set intersections with batched binary search rather than an
   all-pairs comparison;
3. copies compact shared-count and valid-length arrays to pinned host memory
   asynchronously;
4. writes one compressed shard per `(step, layer, rank)` from a background
   thread; and
5. suppresses activation-checkpoint recomputation duplicates with the same
   `(step, layer, rank)` key.

The final data set contains:

- 1,920 raw shards: 10 steps x 24 layers x 8 ranks;
- 131,072 adjacent-query pairs per `(step, layer)`;
- 3,145,728 pair observations per step; and
- 31,457,280 pair observations in total.

No step, layer, or rank shard was missing or duplicated. The training run had no
NaN or skipped iteration.

## 4. Overall overlap by step

The threshold columns report the fraction of all pair observations whose
normalized overlap meets or exceeds the threshold. `Delta` is the change in
mean overlap from the preceding step.

| Step | Mean overlap | Delta | Median | P90 | Mean Jaccard | Mean shared indices | Overlap >= 50% | Overlap >= 75% | Overlap >= 80% |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 13.18% | +0.00 pp | 7.71% | 30.57% | 8.12% | 261.9 | 3.22% | 1.53% | 1.19% |
| 1 | 50.51% | +37.34 pp | 55.18% | 70.51% | 36.15% | 1,026.5 | 62.64% | 5.76% | 3.09% |
| 2 | 62.94% | +12.43 pp | 66.70% | 78.52% | 48.18% | 1,281.1 | 85.43% | 24.37% | 7.21% |
| 3 | 72.98% | +10.04 pp | 76.37% | 85.99% | 59.53% | 1,486.7 | 95.50% | 55.76% | 30.80% |
| 4 | 74.46% | +1.47 pp | 77.25% | 87.21% | 61.37% | 1,516.9 | 95.85% | 62.17% | 34.40% |
| 5 | 72.05% | -2.41 pp | 75.63% | 87.79% | 58.88% | 1,467.6 | 93.37% | 51.77% | 34.91% |
| 6 | 74.68% | +2.63 pp | 78.71% | 87.89% | 61.91% | 1,521.5 | 95.29% | 62.01% | 44.09% |
| 7 | 75.21% | +0.53 pp | 80.52% | 87.84% | 62.83% | 1,532.3 | 94.36% | 68.51% | 52.45% |
| 8 | 75.47% | +0.26 pp | 78.61% | 87.50% | 62.77% | 1,537.7 | 95.86% | 71.18% | 42.91% |
| 9 | 72.50% | -2.97 pp | 77.29% | 86.13% | 59.28% | 1,476.9 | 92.81% | 60.32% | 32.48% |

The largest transition happens immediately: one optimizer update raises the
mean from 13.18% to 50.51%. By step 3, the mean has reached 72.98%, after which
it fluctuates within a few percentage points rather than increasing
monotonically.

The difference between the mean and the threshold distributions matters. At
step 7, for example, the mean is 75.21%, but only 52.45% of pairs reach 80%
overlap. A kernel policy that assumes every pair has approximately the mean
overlap would therefore overestimate its reusable work.

## 5. Layer-depth behavior

| Step | Layer 1 | Layers 1-8 mean | Layers 9-16 mean | Layers 17-24 mean | Layer 24 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.93% | 5.14% | 10.93% | 23.47% | 19.20% |
| 1 | 6.67% | 32.06% | 59.23% | 60.24% | 64.22% |
| 2 | 4.90% | 48.83% | 67.13% | 72.87% | 77.45% |
| 3 | 6.34% | 64.08% | 78.60% | 76.28% | 80.86% |
| 4 | 4.85% | 66.22% | 79.71% | 77.44% | 85.48% |
| 5 | 6.39% | 60.88% | 78.44% | 76.84% | 89.61% |
| 6 | 5.05% | 65.60% | 79.04% | 79.41% | 87.69% |
| 7 | 6.49% | 63.05% | 82.00% | 80.58% | 83.03% |
| 8 | 6.15% | 67.41% | 82.52% | 76.48% | 80.32% |
| 9 | 6.67% | 60.84% | 79.32% | 77.34% | 79.92% |

Depth is a stronger predictor than the model-wide mean during the first few
updates. On step 0, the first eight layers average only 5.14%, while the last
eight already average 23.47%. After training starts, layers 2 through 24 rapidly
develop much higher overlap, but Layer 1 remains a persistent low-overlap
outlier.

At step 9:

- Layer 1 has 6.67% mean overlap;
- Layer 2 has 72.60%;
- Layer 3 has 51.87%;
- the maximum is Layer 13 at 88.27%; and
- 23 of 24 layers exceed 50% mean overlap.

This spread is large enough that a single model-wide q-cluster decision would
be structurally wrong for at least one layer.

## 6. Implications for q-cluster design

### 6.1 There is substantial reusable KV work

Once overlap reaches 70% to 80%, a two-query cluster can often split its sparse
selection into three logical regions:

1. indices shared by both queries;
2. indices unique to the left query; and
3. indices unique to the right query.

The shared region creates an opportunity to amortize index processing and KV
loads across the two query rows. It may also improve locality in the backward
dKV path. These measurements quantify the available overlap; they do not by
themselves prove that every shared operation can be eliminated or that the
end-to-end kernel speedup is proportional to overlap.

### 6.2 A fixed shared-index count is not robust

The observed shared count varies across layers, steps, and query pairs. Even in
the high-overlap regime, the fraction of pairs above 80% changes from 30.80% at
step 3 to 52.45% at step 7 and back to 32.48% at step 9.

A public interface should therefore avoid asking the user to supply a fixed
`shared_topk`. The implementation can instead derive the exact intersection
size during preprocessing and select among a small number of efficient kernel
paths.

### 6.3 Gating should be adaptive and cheap

A practical policy can combine:

- a user-visible high-overlap hint that is disabled by default;
- per-layer historical statistics or a lightweight sampled estimate;
- an exact compact intersection during preprocessing when the hint is enabled;
- a q-cluster path only when the predicted savings exceed preprocessing cost;
  and
- a direct baseline fallback for Layer 1 and other low-overlap inputs.

The gate must be based on end-to-end cost, not overlap alone. A pair with 40%
overlap may still be profitable if preprocessing is almost free, while a pair
with 70% overlap may not help if regrouping and scatter overhead dominate the
saved sparse loads.

### 6.4 CUDA Graph compatibility favors shape-stable outputs

Adaptive overlap does not require dynamic allocation or a graph break. A
preprocessing kernel can write into fixed-capacity buffers sized by top-k,
along with device-side counts describing the shared and two unique regions.
The captured graph and launch topology remain fixed; only device-resident
counts and branch predicates change between replays.

## 7. Limitations and next measurements

This experiment should not be interpreted as the final overlap distribution of
a production Ling checkpoint:

- only 10 optimizer steps were measured;
- the model started from random initialization;
- global batch size was one 256K sequence;
- only two open-data sources were used; and
- the experiment measured index reuse, not q-cluster end-to-end latency.

The next validation should replay the same instrumentation on a converged or
late-training checkpoint and report distributions by layer, dataset, sequence
position, top-k, and q-pair distance. The most useful kernel study would then
measure preprocessing plus attention end to end at several overlap quantiles,
especially around the actual crossover point rather than only at synthetic 0%,
50%, and 100% overlap.

## 8. Main takeaway

The 256K trace shows that adjacent-query KV-index overlap can become both large
and common very early in DSA training. That makes q-cluster a credible
structural optimization. The same trace also shows why the implementation must
remain adaptive: one layer can stay near 6% while neighboring layers exceed
70%, and the high-overlap tail changes materially from step to step.

The right abstraction is therefore not "use exactly N shared indices." It is
"detect the reusable region cheaply, process it once when profitable, and keep
the baseline path effectively free when it is not."
