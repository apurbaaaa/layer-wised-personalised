# Response to Reviewer Comments — Revision 2

Comments 2, 4, 5, 7, 8 and 9 are addressed as in our previous response and are
unchanged. Comments 1, 3 and 6 are rewritten below, because new experiments
were run for this revision. Two methodological issues that we discovered while
running them are disclosed at the end; both affect how our earlier numbers
should be read, and we raise them ourselves rather than leave them implicit.

New experiments (all seed 42, K=5, 25 rounds, identical hyperparameters):

| Arm | Decomposition | Drift weighting | Personalized head | α |
|-----|---------------|-----------------|-------------------|---|
| `full` (FedSA-Drift) | yes | yes | yes | 0.1 |
| `decomp_nodrift` | yes | no | yes | 0.1 |
| `true_fedavg` | no | n/a | no | 0.1 |

All three share an identical Dirichlet partition (same seed), architecture,
round count and hyperparameters, so the only variable is the aggregation
scheme. A fourth `true_fedavg` run at α = 0.5 was also performed and is
reported in the consolidated results table.

All arms were additionally evaluated on the **held-out ISIC 2019 test split**,
which was never used for training or model selection.

---

## Comment 6 — α = 0.1 was never evaluated

**Addressed with new experiments.**

We have now evaluated the α = 0.1 regime at K = 5. The resulting partition is
genuinely severe — one client holds no samples at all for four of the eight
classes, two more are missing three and two classes, and the largest client
holds 11× the data of the smallest:

| Client | Samples | Classes present |
|--------|---------|-----------------|
| 0 | 1015 | 8 |
| 1 | 2687 | 6 (98 % in a single class) |
| 2 | 11339 | 4 |
| 3 | 4945 | 7 |
| 4 | 1545 | 5 |

Comparing full FedSA-Drift against the decomposition-only arm (drift disabled)
under an identical partition, evaluated per client with each client's own
trained head:

| Metric | FedSA-Drift | Decomposition only | Δ |
|--------|-------------|--------------------|---|
| **Validation** | | | |
| Mean per-client accuracy | 0.5018 | 0.3579 | **+14.39 pp** |
| Mean per-client balanced accuracy | 0.4382 | 0.3798 | **+5.84 pp** |
| Worst-client accuracy | 0.4297 | 0.1953 | **+23.44 pp** |
| Std. of per-client accuracy | 0.0551 | 0.1294 | **−57.4 %** |
| **Test (held out)** | | | |
| Mean per-client accuracy | 0.4264 | 0.3111 | **+11.53 pp** |
| Mean per-client balanced accuracy | 0.3681 | 0.3374 | **+3.07 pp** |
| Worst-client accuracy | 0.3852 | 0.1699 | **+21.53 pp** |
| Std. of per-client accuracy | 0.0300 | 0.0911 | **−67.1 %** |

The test split reproduces every finding, with the variance reduction larger
than on validation.

Placing this beside the previously reported settings, the worst-client benefit
is **largest** in the most heterogeneous regime:

| Setting | Worst-client gain | Reduction in std. of accuracy |
|---------|-------------------|-------------------------------|
| α = 1.0, K = 3 | +13.71 pp | −58 % |
| α = 0.5, K = 5 | +9.10 pp | −39 % |
| **α = 0.1, K = 5 (new)** | **+23.44 pp** | **−57.4 %** |

The result is not an artefact of the final round: means over rounds 21–25 give
+23.04 pp worst-client and −56.7 % dispersion, closely matching the round-25
values.

We also note that at α = 0.1 there is **no balanced-accuracy cost** —
FedSA-Drift is ahead by 5.84 pp — in contrast to α = 0.5, where we reported a
utility cost. Overall (unbalanced) accuracy is lower for FedSA-Drift
(0.4787 vs 0.6116 global), consistent with the method trading majority-class
predictions for minority-class recall on a heavily imbalanced dataset.

---

## Comment 3 — The A/B/C decomposition has no ablation

**Addressed with a new three-way experiment.**

We ran the missing configuration: a `true_fedavg` arm at α = 0.1 in which the
decomposition is removed entirely — every tensor aggregated by standard
size-weighted FedAvg, no client-specific head. With the two arms above this
gives a complete three-point ladder under an identical partition, seed and
hyperparameters, every arm evaluated per client with a trained classifier.
Held-out test split:

| Configuration | Mean acc | Mean bal. acc | Worst acc | Std acc | MEL recall |
|---------------|----------|---------------|-----------|---------|------------|
| No decomposition | 0.4702 | 0.3778 | — | — | **0.000** |
| Decomposition only | 0.3111 | 0.3374 | 0.1699 | 0.0911 | 0.167 |
| Full FedSA-Drift | 0.4264 | 0.3681 | 0.3852 | 0.0300 | **0.228** |

The two components turn out to do different jobs, which we consider a more
informative answer than a single scalar contribution.

**The decomposition supplies class coverage.** Standard FedAvg attains the
highest mean accuracy but abandons four of eight classes: recall is exactly
zero for melanoma and SCC and below 0.03 for AK and BCC. Adding the
decomposition restores all eight (MEL 0.000 → 0.167, SCC 0.000 → 0.194,
AK 0.003 → 0.205, BCC 0.028 → 0.378), at a cost of 15.9 pp of mean accuracy.

**Drift-aware weighting supplies utility and fairness.** Added on top, it
returns 11.5 pp of mean accuracy, raises the worst-client floor by 21.5 pp,
reduces inter-client dispersion by 67 %, and improves melanoma recall further
to 0.228.

Neither component is redundant and neither suffices alone. This is reported as
a new Table and subsection in the revised manuscript.

**We state plainly what this does not show.** FedSA-Drift does *not* improve
aggregate accuracy at α = 0.1: standard FedAvg is ahead on both mean accuracy
(0.4702 vs 0.4264) and balanced accuracy (0.3778 vs 0.3681). Our case in this
regime rests on class coverage, worst-client performance and dispersion, and we
have added a Limitations entry saying exactly that.

**A finding we think is of independent interest.** The two configurations
differ by roughly one point of balanced accuracy yet have entirely different
clinical profiles — one is strong on four classes and scores zero on four, the
other is moderate across all eight. Balanced accuracy, being the unweighted
mean of per-class recalls, cannot distinguish them. Under standard FedAvg,
melanoma recall is zero for *every* client, including the client holding 3815
of the federation's 3844 melanoma cases, because size-weighted aggregation is
dominated by a client holding 53 % of the data and no melanoma at all. We
therefore argue that per-class coverage should be reported as a primary
criterion in federated medical imaging rather than as a diagnostic
afterthought.

---

## Comment 1 — Single-run point estimates without confidence intervals

**Addressed for evaluation uncertainty, which now carries interval estimates;
training-run uncertainty is retained as an explicit limitation.**

On reflection the original criticism conflates two distinct sources of
uncertainty, and separating them lets us address one rigorously and be precise
about the other.

**(a) Evaluation-set sampling uncertainty — now quantified.** Per-class recall
is a binomial proportion (*k* correct out of the *n* test cases of that class),
so it admits an exact-coverage confidence interval from a single trained model,
with no repeated training. We now report 95 % Wilson score intervals on the
held-out test split. For the melanoma result, on the client holding 3815 of the
federation's 3844 melanoma cases:

| Configuration | MEL recall | 95 % CI |
|---------------|-----------|---------|
| No decomposition | 0.000 | [0.000, 0.003] |
| Decomposition only | 0.573 | [0.546, 0.599] |
| Full FedSA-Drift | 0.727 | [0.703, 0.750] |

The three intervals are mutually disjoint, so this ordering is not an artefact
of evaluation-sample noise. An observed 0/1327 places the upper bound at 0.003:
the failure of standard FedAvg to detect melanoma is not a small-sample
accident. The corresponding figure for SCC is 0/165, upper bound 0.023. Overall
test accuracy for that arm is 0.4702, 95 % CI [0.4578, 0.4826], on n = 6191.
This is now Section IV-E.3 and Table 5 of the manuscript.

**(b) Training-run (seed) uncertainty — not quantified, and stated as such.**
Seed-to-seed variability requires repeating each configuration across multiple
random partitions and initialisations. At approximately 10 GPU-hours per run
this remained beyond the compute available for this revision. The figures the
Reviewer names specifically — 58 % variance reduction, +13.71 pp worst-client
accuracy — are affected by this source and remain single-trial point estimates.
We make no claim of statistical significance for them.

**What we have done short of a multi-seed study.** Seeding is now an explicit,
logged experimental parameter (it was previously a fixed internal constant, so
every reported run silently shared one partition and one initialisation, which
we consider a reporting deficiency in its own right and have corrected). The
α = 0.1 results are confirmed on a held-out test split never used for training
or model selection. And we verify that those effects are stable across the
final five communication rounds rather than resting on a single round: means
over rounds 21–25 give +23.04 pp worst-client and −56.7 % dispersion against
final-round values of +23.44 pp and −57.4 %.

We note explicitly that neither round-to-round stability nor the evaluation
intervals is a substitute for a multi-seed study, and the revised Limitations
section distinguishes the two forms of uncertainty rather than implying the
interval estimates cover both. We would rather report a transparently-scoped
result than present an underpowered two-seed range as though it were a
confidence interval over training runs.

---

## Disclosure 0 — Same-lesion contamination in the validation split

While preparing this revision we audited our train/validation split against the
ISIC 2019 `lesion_id` metadata and found a problem we had not previously
recognised, which we report here in full.

The archive assigns a `lesion_id` to 23,247 of its 25,331 images, covering
11,847 distinct lesions, so most lesions contribute several photographs. Our
split is stratified over *images*, not lesions. Auditing it: **1,881 lesions
have images on both sides, and 2,340 of the 3,800 validation images (61.6 %)
depict a lesion that also appears in the training partition.** No image occurs
in both splits, but sibling images of the same lesion do.

Validation-split figures are therefore optimistically biased. This includes the
centralized 0.8867 and every round-by-round figure at α ∈ {1.0, 0.5}. The scale
of the bias is visible in the gap to the held-out test collection, where
balanced accuracy falls to 0.368–0.378.

We have restructured the manuscript in response rather than merely annotating
it:

- The abstract now leads with the held-out test result, not the centralized
  figure.
- The centralized run is presented explicitly as a reference ceiling under
  matched data conditions, not as a generalization estimate, and we no longer
  draw any margin against externally reported numbers.
- The Summary of Results orders findings by evidence strength and states which
  are test-split, which are within-split differences, and which are absolute
  validation figures affected by the contamination.
- A Limitations entry gives the audit numbers.

Two classes of result are unaffected, and these are the ones our claims rest
on. Comparisons *between* configurations remain valid because every arm is
trained and evaluated on the identical split, so the contamination is common to
all and cancels in the differences — worst-client accuracy, dispersion, and the
component ablation. And every test-split result, including the entire three-way
ablation and the melanoma finding, is computed on the separate ISIC 2019 test
collection.

A lesion-grouped split is the correct design and would yield unbiased absolute
validation figures; it requires retraining every configuration and is stated as
future work. We would rather surface this ourselves than have it found.

## Disclosure 1 — Global-model accuracy understates personalized methods

While preparing this revision we verified empirically that the **global model
does not contain a trained classifier** in any arm that uses the
decomposition. Group C comprises `metadata_mlp` and `fusion_head`; because the
backbone is instantiated with `num_classes=0`, `fusion_head` *is* the
classifier. Group C is never aggregated, so the global model retains its random
initialisation for the entire run. We confirmed this directly: after training,
all 8 Group C tensors are bit-identical to their initialisation
(max |Δ| = 0.000), while Group A tensors have changed.

Consequences, stated precisely:

1. **Our fairness results are unaffected.** Worst-client accuracy and
   inter-client dispersion were always computed by evaluating each client with
   its own trained head. Every fairness claim in the paper stands as reported.
2. **Global-accuracy figures for decomposed arms are a lower bound.** They
   measure a model with an untrained classifier and therefore *understate*
   achievable performance. They remain valid for comparisons *within* the
   decomposed family (FedSA-Drift vs. decomposition-only), where both arms
   carry the same handicap.
3. **Comparisons across families are not valid.** `true_fedavg` has an empty
   Group C, so its classifier is aggregated and trained. Its global accuracy
   (0.8260 balanced, α = 0.5) cannot be compared with the decomposed arms'
   global accuracy. Likewise, the comparison of federated global accuracy
   against centralized training, and against published methods, compares a
   model without a trained head to models with one.

Accordingly we now report **mean per-client accuracy** as the primary utility
metric, which is the standard choice in the personalized federated learning
literature and the only accuracy metric comparable across all arms. We have
softened the corresponding claims in the abstract and in Section IV-C, and
added this to the Limitations section. A further symptom, which we report as
confirmation: `true_fedavg` yields an inter-client standard deviation of
exactly 0.0000 in all 25 rounds, since without a personalized head every
client evaluates an identical model.

## Disclosure 2 — The dispersion result depends on the accuracy measure

At α = 0.1, dispersion across clients moves in opposite directions depending on
which accuracy is used:

| Dispersion measure (val) | FedSA-Drift | Decomposition only |
|--------------------------|-------------|--------------------|
| Std. of per-client accuracy | **0.0552** | 0.1294 |
| Std. of per-client balanced accuracy | 0.0855 | **0.0587** |

Our reported variance-reduction figures use the standard deviation of overall
accuracy, consistent with the earlier revisions. On balanced accuracy the
ordering reverses, and we report this rather than omit it.

The mechanism is not that any client is harmed. Every client's balanced
accuracy improves under drift-aware aggregation, but by unequal amounts, and
the magnitude tracks class coverage:

| Client | Classes present | BalAcc (drift) | BalAcc (no drift) | Gain |
|--------|-----------------|----------------|-------------------|------|
| 0 | 8 | 0.521 | 0.428 | +9.3 pp |
| 3 | 7 | 0.526 | 0.440 | +8.6 pp |
| 4 | 5 | 0.473 | 0.413 | +6.0 pp |
| 2 | 4 | 0.340 | 0.307 | +3.3 pp |
| 1 | 6 (98 % one class) | 0.332 | 0.310 | +2.2 pp |

Clients holding seven or eight classes gain roughly 9 pp; the client holding
four gains 3 pp. Balanced accuracy is bounded by classes a client has never
observed, so drift-aware aggregation cannot lift those clients as far. The
dispersion therefore widens *because* the benefit is unevenly distributed, not
because performance degrades anywhere. On the test split the same pattern
holds, with four clients improving and one unchanged within noise (−0.4 pp).

We accordingly state the fairness claim precisely: drift-aware aggregation
raises the worst-client floor and reduces dispersion **in overall accuracy**,
while improving every client's balanced accuracy by an amount that scales with
that client's class coverage.
