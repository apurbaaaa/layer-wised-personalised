"""
Fix hallucinations and broken sentences in FedVitHuman.docx.
Minimal edits - preserve the humanized voice, only fix what doesn't make sense.
"""
from docx import Document

DOCX_PATH = "FedVITpaper/FedVitHuman.docx"

# Each tuple: (substring_to_find, replacement)
# Using substring matching so we can target individual sentences within paragraphs.
REPLACEMENTS = [
    # ---- ABSTRACT ----
    # "facilitates the sharing of heterogeneous medical data" — FL does NOT share data.
    (
        "In addition to being able to comply with privacy regulations such as GDPR and HIPAA, federated learning also facilitates the sharing of heterogeneous medical data among different partner organizations.",
        "In addition, federated learning supports compliance with privacy regulations such as GDPR and HIPAA, while enabling multi-institutional collaboration on heterogeneous medical data without sharing raw records.",
    ),
    (
        "The diversity in the medical data collected by organizations often leads to varied levels of data quality between clients, which can lead to a number of challenges during the training process, including client drift, unstable convergence, and fairness.",
        "The diversity in the medical data collected by different organizations often leads to heterogeneity between clients, which can cause a number of challenges during the training process, including client drift, unstable convergence, and fairness issues.",
    ),
    # Wrong group decomposition + hallucinated (ID-AGG) abbreviation + wrong "from the federated-averaging"
    (
        "These challenges can become more pronounced when training multimodal (structured and unstructured) medical data sources (e.g., clinically relevant metadata, as well as the high-capacity Vision Transformer models used to analyze those sources). This paper proposes a framework called FedSA-Drift (Structure-Aware Drift Regularized Federated Learning) for classifying skin lesions using multimodal medical data. FedSA-Drift features a SwinV2-Base backbone that is fused with the structured metadata for the skin lesions, and it separates the parameters of the SwinV2-Base into three different categories: 1) early vision layers (for image input), 2) deep Transformer layers (for structured metadata), and 3) client-specific multimodal heads (for personalized outputs). The framework also introduces drift-aware inverse-distance aggregation (ID-AGG) to reduce client divergence from the federated-averaging federated learning algorithm without modifying local objectives or adding any additional hyperparameters to the local model.",
        "These challenges can become more pronounced when training on multimodal medical data, where structured clinical metadata is combined with high-capacity Vision Transformer models. This paper proposes a framework called FedSA-Drift (Structure-Aware Drift-Regularized Federated Learning) for classifying skin lesions using multimodal medical data. FedSA-Drift features a SwinV2-Base backbone fused with structured metadata, and it separates the parameters into three different categories: 1) early vision layers, 2) deeper Transformer layers, and 3) client-specific multimodal heads. The framework also introduces a drift-aware inverse-distance aggregation method to reduce client divergence, without modifying local objectives or adding any additional hyperparameters.",
    ),

    # ---- INTRO ----
    # "disproportional amount of deaths" → "disproportionate number of deaths"
    (
        "MELANOMA is responsible for disproportional amount of deaths from skin cancer.",
        "MELANOMA is responsible for a disproportionate number of deaths from skin cancer.",
    ),
    # "Client metadata" → "Patient metadata"
    (
        "Client metadata contains information about the clinical context that cannot be derived from visual features only",
        "Patient metadata contains information about the clinical context that cannot be derived from visual features only",
    ),
    # Broken sentence: "client drift, and aggregating both models must be handled differently"
    (
        "Therefore, in order to build a stable and equitable multimodal federated system, client drift and aggregating both models must be handled differently.",
        "Therefore, in order to build a stable and equitable multimodal federated system, client drift must be addressed and the structural differences between model components must be respected.",
    ),

    # ---- LITERATURE SURVEY ----
    # "using mask the patches" typo + "NN IID" hallucination
    (
        "Strategies to increase efficiency include using mask the  patches for training [28] and pruning layers in the ViT architecture [29]. In addition to being adversely affected by NN IID data, feature collapse is also an issue with Federated ViT learning.",
        "Strategies to increase efficiency include masking patches during training [28] and pruning layers in the ViT architecture [29]. In addition to being adversely affected by non-IID data, feature collapse is also an issue with Federated ViT learning.",
    ),
    # "partial-order personalization" → "partial personalization"
    (
        "In multimodal systems, partial-order personalization leads to better adaptation at the local client",
        "In multimodal systems, partial personalization leads to better adaptation at the local client",
    ),
    # "computations will be needed twice, plus 1 for any dual variable update(s)" - nonsense
    (
        "The additional complexity of optimization-based algorithms will also add to this and the overall amount of computations will be needed twice, plus 1 for any dual variable update(s). On the other hand, alignment is performed at a penalty proportional to the total dimension of the attention for each part of the entire set of parameters: typically,",
        "Optimization-based approaches further increase this cost due to additional penalty terms and dual-variable updates. Alignment-based methods also introduce overhead proportional to the attention dimensions, typically",
    ),

    # ---- METHODOLOGY ----
    # Duplicated DIRICHLET
    (
        "C. 1. Step 1: (Dirichlet) Non-IID Data Partition (DIRICHLET)",
        "C. STEP 1: DIRICHLET NON-IID DATA PARTITIONING",
    ),
    # "submits the low-level visual features..." + "differentially shocked weighting scheme" hallucination
    (
        "G = {GA, GB, GC,k} (i.e. group A [GA] submits the low-level visual features that will generalize to the different institutions to the standard FedAvg aggregation, group B [GB] submits the higher-level visual representations (i.e. Swin-V2) produced in the final two layers (i.e. the third and fourth), and the weights are submitted with a differentially shocked weighting scheme to assist with detecting drift) and group C [GC,k] - the client-specific metadata MLP (multilayer perceptron) and the fusion head are never transmitted to the server.",
        "θ = {θA, θB, θC,k}, where Group A (θA) contains the patch embedding and early SwinV2 stages (0–1) that capture low-level visual features generalising across institutions, aggregated by standard FedAvg; Group B (θB) contains the late SwinV2 stages (2–3) and layer normalisation that encode high-level semantics most sensitive to distribution shifts, aggregated with drift-aware weighting; and Group C (θC,k) contains the client-specific metadata MLP and fusion head, which are never transmitted to the server.",
    ),
    # "the inferences for Groups A and B will be uploaded"
    (
        "Once local training is complete, the inferences for Groups A and B will be uploaded to the server",
        "Once local training is complete, the parameters for Groups A and B will be uploaded to the server",
    ),
    # Wrong variable "Pk,l" instead of "Dk,l"
    (
        "∝k,l=(1/(Pk,l+ε))",
        "αk,l = 1/(Dk,l + ε)",
    ),
    (
        "The inverse approach is preferred to an exponential, e.g., exp(-βPk,l)",
        "The inverse approach is preferred to an exponential, e.g., exp(-βDk,l)",
    ),
    # "convex losses" hallucination
    (
        "For local convex losses with an L-smooth condition",
        "For L-smooth local objectives",
    ),
    # "When compared the computational structure" - awkward
    (
        "When compared the computational structure to a centralised system (Algorithm 1), the implementation structure is completely different for the federated system (Algorithm 2). In the centralised training process (as used by a single organisation), the optimisation is based on the entire collection of pooled data – however, in the federated training process with on-site training of clients and obtaining a global consensus through the back-bone of the server, the parameters that the clients create using their on-site data are never sent to the server. This structure preserves the privacy of institutional metadata as well as provide capability for institutional specialisation.",
        "Compared to the centralized system (Algorithm 1), the federated implementation (Algorithm 2) is structurally different. In centralized training, optimization runs directly on pooled data, while in federated training, clients perform on-site training and the server enforces a global consensus over the shared backbone. The Group C parameters that clients create using their on-site data are never sent to the server. This structure preserves the privacy of institutional metadata while still providing capability for institutional specialisation.",
    ),
    # "is to be evaluated" tense
    (
        "FedSA-Drift is to be evaluated on the ISIC 2019 dataset",
        "FedSA-Drift is evaluated on the ISIC 2019 dataset",
    ),
    # "samples are essentially from the same population (i.e., IID)"
    (
        "For instance, the case when α = 1 simulates the case where the samples are essentially from the same population (i.e., IID). When α = 0.5, we simulate unequal distributions of samples between class labels.",
        "For instance, α = 1.0 simulates near-IID conditions, while α = 0.5 simulates highly heterogeneous class distributions across clients.",
    ),
    # Learning rate hallucination - "0.1 for the SwinV2 backbone" should be "0.1η"
    (
        "The learning rate (η) is set to be 0.1 for the SwinV2 backbone, while η=4×10−4 is used for the MLP that predicts the metadata and the fusion head.",
        "A differential learning rate is applied. The SwinV2 backbone uses 0.1η, while the metadata MLP and fusion head use η = 4×10⁻⁴.",
    ),

    # ---- RESULTS ----
    # "saturation period with respect to performance" - awkward
    (
        "Epoch 21 was end of saturation period with respect to performance demonstrating convergence.",
        "Performance saturated after Epoch 21, demonstrating convergence.",
    ),
    # Broken section header merged with paragraph
    (
        "C. Mitigating exacerbated drift and ensuring client fairness are important for federated multi-modal networks that keep the metadata heads local but share a common backbone that can create increased client drift. The visual backbone also will be driven toward different local optimal points. This has been confirmed in the experiments we've conducted with FedSA-Drift and is described in more detail below.",
        "C. MITIGATING EXACERBATED DRIFT AND ENSURING CLIENT FAIRNESS\nIn multimodal federated networks that keep the metadata heads local but share a common backbone, client drift can be exacerbated as the visual backbone is driven toward different local optima. We have confirmed this in our experiments with FedSA-Drift and describe it in more detail below.",
    ),
    # "(approx. 8.8 out of 10)" nonsense + "drift-driven class weighting"
    (
        "The K = 3 clients have almost equal proportions of the classes that have been distributed across multiple clients with α = 1.0 (approx. 8.8 out of 10). Two different types of aggregate methodologies were compared across these K = 3 clients: 1) A standard aggregated Fed Avg, and 2) A FedSA-Drift that used drift-driven class weighting based on the distribution of the classes on Group B.",
        "With K = 3 clients and α = 1.0, the class distributions are nearly uniform across clients. Two different aggregation methodologies were compared: 1) standard FedAvg, and 2) FedSA-Drift that used drift-aware weighting on Group B.",
    ),
    # "Under these mild levels of client hétérogénity"
    (
        "Under these mild levels of client hétérogénity, Fed Avg only achieved a balanced global accuracy rate of 0.8314; however, the client with the lowest accuracy rate dropped down to 0.6479, which would not be apparent from the global average. Using FedSA-Drift, the client with the lowest accuracy rate saw his accuracy increase by +13.71% to a value of 0.7850, and the overall standard deviation of client-to-client differences decreased by 58% from 0.0182 to 0.0076, with approximately 93-94% of both Fed Avg and FedSA-Drift clients achieving global centralized performance.",
        "Under these mild levels of client heterogeneity, FedAvg achieved a balanced global accuracy of 0.8314; however, the worst-performing client dropped to 0.6479, which would not be apparent from the global average. Using FedSA-Drift, the worst-client accuracy increased by +13.71 points to 0.7850, and the inter-client standard deviation decreased by 58% from 0.0182 to 0.0076. Both methods retain 93–94% of centralized performance globally.",
    ),
    # "Federal-Self-Attention Drift" hallucinated expansion + "minimum weight of each drift layer"
    (
        "Federal-Self-Attention Drift (FedSA-Drift) achieved comparable attention stability by using Group A with FedAvg and using the minimum weight of each drift layer at the server. There was no modification to the client-side objective function. The balanced accuracy of 82.09% attained by FedSA-Drift at 𝛼 = 1.0 is competitive with that of FedMHA, which is 83.99% at 𝛼 = 0.9.",
        "FedSA-Drift achieves comparable attention stability by aggregating Group A with FedAvg and down-weighting deep-layer drift at the server. There was no modification to the client-side objective function. The balanced accuracy of 82.09% attained by FedSA-Drift at α = 1.0 is competitive with that of FedMHA, which is 83.99% at α = 0.9.",
    ),
    # "In lieu of an alternate solution... FedSA-Drift provided total spatial representations" — broken
    (
        "In lieu of an alternate solution to totally mitigate drift at the server and achieve no client-side impact, FedSA-Drift provided total spatial representations.",
        "FedSA-Drift maintains full spatial representations and achieves drift mitigation purely at the server with zero client-side cost.",
    ),
    # "FedAPM obtained superior performance over multimodal metrics"
    (
        "Through this architecture, FedAPM obtained superior performance over multimodal metrics; however, the amount of inner-loop iterations required for every ADMM sub-problem were generally prohibitive based on the general parameters of SwinV2-Base.",
        "Through this approach, FedAPM achieves strong performance on multimodal benchmarks; however, the number of inner-loop iterations required for each ADMM sub-problem is generally prohibitive given the parameter count of SwinV2-Base.",
    ),
    # "(2019)" wrong year for Yaqoob
    (
        "According to Yaqoob et al. (2019),",
        "Yaqoob et al. [25] reported",
    ),
    (
        "however, these metrics are misleading on a class-imbalanced dataset. The ISIC 2019 dataset standard metric is balanced accuracy, and this is a more appropriate metric for this analysis. In addition, the CNN architecture used by Yaqoob et al. (2019)",
        "however, these metrics are misleading on a class-imbalanced dataset. The ISIC 2019 dataset standard metric is balanced accuracy, and this is a more appropriate metric for this analysis. In addition, the CNN architecture used by Yaqoob et al.",
    ),
    # "completed visit to the server" - nonsense word "visit"
    (
        "and the Inverse Distance Weighting are completed visit",
        "and the Inverse Distance Weighting are completed at",
    ),
    (
        "the server on O( |wshared|) linear calculations",
        "the server in O(|wshared|) linear operations",
    ),
    # "FedSA-Drift will impact a maximum of 11 seconds per round." - awkward phrasing
    (
        "FedSA-Drift will impact a maximum of 11 seconds per round.",
        "FedSA-Drift adds at most 11 seconds per round.",
    ),
    # "Completely balanced accuracy convergence for each of the tests con-ducted as depicted in Figure 11."
    (
        "Completely balanced accuracy convergence for each of the tests con-ducted as depicted in Figure 11.",
        "Figure 11 shows balanced accuracy convergence across all configurations.",
    ),
    # "FedSA-Drift provides an equivalent level of overall drift and fairness to beat any system currently proposed"
    (
        "Consequently, FedSA-Drift provides an equivalent level of overall drift and fairness to beat any system currently proposed (FedMHA and FedAPM) without adjusting local objectives or adding hyperparameters. Thus, global accuracy only is insufficient for measuring federated medical models. However, FedSA-Drift does offer a practical path",
        "Consequently, FedSA-Drift matches the drift mitigation and fairness of currently proposed systems (FedMHA and FedAPM) without adjusting local objectives or adding hyperparameters. Thus, global accuracy alone is insufficient for evaluating federated medical models. FedSA-Drift offers a practical path",
    ),

    # ---- CONCLUSION ----
    # "selective cross-session aggregation" hallucination
    (
        "thus allowing for both selective cross-session aggregation and individual personalization",
        "thus allowing for both selective aggregation and individual personalization",
    ),
    # Run-on sentence with "structured quantization... cross-dataset and domain shift comparisons"
    (
        "In addition, using structured quantization to achieve communication efficiency could potentially allow for the implementation of a large model across many resource-constrained edge devices, therefore producing a much higher level of generalization and result from cross-dataset and domain shift comparisons with respect to assessing the clinical robustness of the model.",
        "In addition, using structured quantization to achieve communication efficiency could potentially allow the implementation of a large model across many resource-constrained edge devices. Cross-dataset generalization and domain shift remain open questions for assessing the clinical robustness of the model.",
    ),
]


def fix_docx():
    doc = Document(DOCX_PATH)
    applied = 0
    not_found = []

    for old, new in REPLACEMENTS:
        replaced_in_this_pass = False
        for para in doc.paragraphs:
            if old in para.text:
                # Replace text - we have to handle runs
                # Simplest approach: replace text in the first run that contains the start
                # and clear the rest. But this loses formatting.
                # Better: rebuild the paragraph text by setting the first run's text and
                # clearing others.
                full_text = para.text
                new_full = full_text.replace(old, new)
                if para.runs:
                    para.runs[0].text = new_full
                    for run in para.runs[1:]:
                        run.text = ""
                applied += 1
                replaced_in_this_pass = True
                break
        if not replaced_in_this_pass:
            not_found.append(old[:80])

    doc.save(DOCX_PATH)
    print(f"Applied {applied} replacements")
    if not_found:
        print(f"\nNot found ({len(not_found)}):")
        for s in not_found:
            print(f"  - {s!r}")


if __name__ == "__main__":
    fix_docx()
