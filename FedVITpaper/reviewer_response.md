# Response to Reviewer Comments

Dear Dr. Arun Kumar Sangaiah,

Thank you for your detailed review. Below is our point-by-point response to each comment.

---

**Comment 1 — Experimental scale too small (K=3,5 only, no confidence intervals)**

We acknowledge this limitation. Unfortunately, we have exhausted our GPU compute credits and cannot run additional experiments at this time. We have added a dedicated **Limitations section** explicitly stating that results are point estimates from single trials without confidence intervals, and that validation at K ∈ {10, 20} remains future work.

---

**Comment 2 — Baseline comparisons are unfair (Table 3, different datasets)**

Addressed. We renamed Section IV-C to **"Contextual Reference Against Related Methods"** and added a prominent disclaimer in both the section text and table caption stating that direct numerical comparison is not appropriate, as methods differ in dataset, backbone, and experimental setup.

---

**Comment 3 — No ablation studies**

Addressed. We added a new **Section IV-E: Component Analysis** with a dedicated **Table 4** comparing drift weighting on vs. off across both configurations, with all other settings held fixed. This isolates the contribution of drift-aware aggregation. A full decomposition ablation requires additional GPU runs and is noted as future work.

---

**Comment 4 — Theoretical stability analysis is circular, not a real proof**

Addressed. We renamed the section to **"Convergence Analysis (Informal Argument)"**, added explicit L-smoothness assumptions, removed the circular σ² inequality, corrected the descent lemma application, and added a clear statement that a formal convergence rate derivation is left to future work.

---

**Comment 5 — Global accuracy trade-off not justified**

Addressed. We expanded Section IV-B.3 to explicitly acknowledge the 7.3 pp gap at α=0.5 is a genuine cost, and grounded the fairness–utility trade-off in the minimax fairness literature (Mohri et al., ICML 2019; Li et al., ICML 2021). We also state that comprehensive clinical validation is future work.

---

**Comment 6 — Heterogeneity range too narrow (α=0.1 never tested)**

We agree α=0.1 is the most challenging regime. We are unable to run this experiment due to exhausted GPU credits. We have added this to the **Limitations section**, renamed the contribution bullet from "Comprehensive Evaluation" to **"Proof-of-Concept Heterogeneity Study"**, and updated the abstract and conclusion to use "moderately heterogeneous" rather than "strongly non-IID" for α=0.5.

---

**Comment 7 — Single dataset, generalization unverified**

Acknowledged. We cannot add cross-dataset experiments at this time due to exhausted GPU credits. The **Limitations section** now explicitly states this, and the abstract is scoped to "a single dataset."

---

**Comment 8 — Single GPU simulation, no communication cost reported**

Addressed. We added **Section IV-D.2: Communication Cost** reporting ~174 MB upload + 174 MB download per client per round (FP16), with the full calculation shown. The Limitations section also notes that real-world network latency and edge hardware constraints are not captured in the simulation.

---

**Comment 9 — SOTA comparison unfair (different models, different datasets)**

Addressed together with Comment 2. The table is now clearly framed as a contextual reference, not a controlled comparison. The only numerically valid comparisons are the internal FedAvg vs. FedSA-Drift results in Tables 1 and 4.

---

We hope these revisions adequately address your concerns. For Comments 1, 6, and 7, we have been transparent about the compute constraints rather than leaving the limitations implicit, and we believe the added Limitations section strengthens the scientific integrity of the paper.

Sincerely,
Apurba Koirala, Rajeshkannan Regunathan, and RA K Saravanaguru
Vellore Institute of Technology, Vellore, India
