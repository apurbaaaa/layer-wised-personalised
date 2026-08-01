# Cover Letter — Revised Submission

**Subject:** Revised manuscript — FedSA-Drift: Structure- and Drift-Aware Federated Vision Transformers for Multimodal Skin Lesion Classification

---

Dear Dr. Sangaiah,

Thank you for the detailed and constructive reviews. They identified several
problems we had not seen, including two factual errors and one methodological
issue in our own data handling that we would not have found otherwise. The
manuscript has been substantially revised. Below we summarise what changed and,
equally important, what we were unable to change.

We should state at the outset that the largest remaining gaps — multiple random
seeds, additional federated baselines, and larger client counts — all require
GPU compute beyond what was available to us. We have not attempted to disguise
these as addressed. They are stated as limitations in the manuscript and
discussed at the end of this letter.

## New experiments

We ran a three-way component ablation at α = 0.1, K = 5, on an identical
Dirichlet partition, and evaluated every configuration on the held-out ISIC
2019 test collection:

1. **No decomposition** — every tensor aggregated by standard FedAvg
2. **Decomposition only** — the A/B/C split, Group B by plain FedAvg
3. **Full FedSA-Drift** — as above, with drift-aware weighting on Group B

This addresses the request for an ablation isolating the parameter
decomposition, and for evaluation under genuinely severe heterogeneity.

## Principal new finding

Under severe heterogeneity, standard FedAvg reaches **zero melanoma recall for
every client**, including the institution holding 3,815 of the federation's
3,844 melanoma cases, because size-weighted aggregation is dominated by a client
holding 52.7 % of the data and no melanoma at all. It nevertheless attains the
highest mean per-client accuracy of the three configurations (0.4702). The
decomposition restores all eight classes; drift-aware weighting then recovers
11.5 points of accuracy, raises the worst-client floor by 21.5 points, and
reduces the inter-client standard deviation by 67 %.

Balanced accuracy differs by roughly one point between the two (0.3778 against
0.3681) despite these very different clinical profiles. We believe the
implication — that aggregate metrics, balanced accuracy included, do not reveal
complete class abandonment — is of interest beyond our particular algorithm, and
we have promoted it to the first contribution.

## Confidence intervals

Per-class recall is a binomial proportion, so it admits exact-coverage intervals
from a single trained model. We now report 95 % Wilson score intervals on the
test split. For melanoma recall on the client holding the melanoma cases, the
three configurations give 0.000 [0.000, 0.003], 0.573 [0.546, 0.599] and 0.727
[0.703, 0.750] — mutually disjoint. An observed 0/1327 places the upper bound at
0.003, so the failure of FedAvg to detect melanoma is not a small-sample
artefact.

We are explicit that these intervals quantify evaluation-set sampling only, not
seed-to-seed variability.

## Data handling — two corrections we raise ourselves

**Split accounting.** The manuscript previously described clients as receiving
all 25,331 images. They receive the 21,531-image training partition; 3,800 are
held out for validation. A new Data Accounting subsection states the arithmetic
in full and reconciles it per class (melanoma: 3,844 + 678 = 4,522, matching the
archive). The ISIC 2019 test collection is a separate set of 8,238 images, of
which 2,047 carry the `UNK` label outside our eight-class space and are
excluded, leaving 6,191.

**Same-lesion contamination.** Auditing our split against the `lesion_id`
metadata, we found that 1,881 lesions have images on both sides, so 2,340 of the
3,800 validation images (61.6 %) depict a lesion also present in training. No
image is duplicated, but sibling images of the same lesion are. Validation
figures — including the centralized 0.8867 — are therefore optimistically
biased. We have restructured the paper rather than merely annotating it: the
abstract now leads with the held-out test result, the centralized run is
presented as a reference ceiling rather than a generalization estimate, and the
Summary of Results states which findings are test-split, which are within-split
differences unaffected by the contamination, and which are absolute validation
figures that it inflates.

## Corrections to claims

- The explanation of the shared global model's classifier was wrong in the
  previous revision. Group C does remain at initialisation, but the model is not
  degenerate: the backbone co-adapts to that fixed projection, and the shared
  model in fact scores higher than the mean personalized client (0.5196 against
  0.4382). Our earlier "lower bound" characterisation was incorrect and has been
  removed.
- Claims of parity with FedMHA and FedAPM, and of comparable attention
  stability, have been removed. We never measured either.
- The comparison against Gessert et al. has been removed entirely. Our
  centralized figure is measured on a contaminated internal validation split and
  is not comparable with results on the official test collection.
- Section III now states explicitly that Equation (27) does not follow for the
  algorithm as implemented, since the Group B rule replaces size weights with
  normalised inverse-distance weights.
- Dispersion percentages are standard-deviation reductions throughout; the paper
  previously used "variance" interchangeably, which is wrong by definition.

## Positioning and analysis

A new subsection places the method against FedPer, FedRep, FedBABU, FedBN,
FedProx, SCAFFOLD, FedNova and FedDyn. We state that neither ingredient is new,
that what is genuinely new is narrow, and that we have not measured against any
of them.

We also added a formal treatment of the weighting scheme: a concentration bound
showing that a client at consensus can absorb essentially the entire aggregate
as D_min → 0; an explicit statement that discarding n_k means the aggregate no
longer estimates the pooled empirical risk, with the bias written out; and an
analysis of the distance metric showing that ‖w_k − w̄‖ = ‖Δ_k − Δ̄‖ (the
broadcast point cancels), so Euclidean distance already measures update
deviation, whereas cosine similarity on raw parameters is near-degenerate.

Computational cost is now reported analytically rather than as wall-clock alone:
parameter counts per group, 173.8 MB per client per round in FP16, and server
aggregation at 0.175 GFLOPs against approximately 7,124 TFLOPs of local training
per round.

## What we could not address

The following require GPU compute we do not have. Each 25-round configuration
costs approximately 10 GPU-hours, and the revisions above already consumed the
compute budget available to us. We would rather state this plainly than present
partial results as sufficient.

- **Multiple random seeds.** All results remain single-trial point estimates. A
  properly powered multi-seed study across the configurations of interest was
  not affordable, and we make no claim of statistical significance for the
  seed-dependent figures. This is stated in the Limitations.
- **Additional baselines.** FedProx, SCAFFOLD, FedBN, FedRep, FedPer, FedBABU,
  FedNova and FedDyn were not evaluated. We identify this as the most
  consequential gap in our evaluation and acknowledge that part of our benefit
  may come simply from not sharing the classification head, as FedPer and FedRep
  also do.
- **Larger client counts.** K ∈ {10, 20, 50} was not run.
- **Aggregation design alternatives.** Cosine-on-updates, softmax, exponential
  and inverse-square weightings, and alternative decomposition boundaries, were
  not evaluated empirically.
- **A lesion-grouped split**, which would give unbiased absolute validation
  figures, requires retraining every configuration.

We recognise that these limit the strength of the conclusions and have adjusted
the claims accordingly throughout, including in the title, abstract and
contribution list.

## Other changes

The title has been shortened and "drift-regularized" replaced with
"drift-aware", since no penalty term is added to any local objective.
Contributions are now separated into algorithmic and empirical. A Reproducibility
section gives the public repository, the seed, and the deterministic recipe for
regenerating the exact splits. References have been reordered
sequentially by first appearance. A Limitations entry has been added on the
absence of calibration and uncertainty analysis.

We hope the revised manuscript is closer to publishable form. We are conscious
that it now discloses more about its own weaknesses than the original did, and
we think it is a better paper for it.

Sincerely,

Apurba Koirala, Rajeshkannan Regunathan, and RA K Saravanaguru
School of Computer Science and Engineering
Vellore Institute of Technology, Vellore, India
