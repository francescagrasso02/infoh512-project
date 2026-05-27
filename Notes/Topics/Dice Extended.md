

## Difference beetween [[DiCE]]
The main difference between the original **DiCE** and the newly introduced **DiCE-Extended** lies in the addition of **robustness** as a core optimization pillar alongside proximity and diversity. While standard DiCE focuses on providing multiple alternative paths for a user, DiCE-Extended ensures those paths remain stable even if there are small variations in the input data.

Here is a detailed breakdown of the differences based on the sources:

### 1. Core Objectives and Metrics

- **DiCE (Diverse Counterfactual Explanations):** The original framework was designed to address two primary challenges: **diversity** (offering many different options to the user) and **feasibility** (ensuring suggestions are realistic). It uses an optimization problem that balances proximity to the original input with a diversity metric based on **Determinantal Point Processes (DPP)**.
- **DiCE-Extended:** This version introduces a third critical objective: **Robustness**. The researchers found that standard DiCE can be sensitive to small perturbations, meaning minor data changes could result in wildly different explanations. To fix this, DiCE-Extended uses the **Dice–Sørensen coefficient** to measure and optimize the stability of counterfactuals under input variations.

### 2. The Loss Function

The mathematical approach to finding counterfactuals has been refined in DiCE-Extended to include a new weight:

- **DiCE:** Balances proximity $(\lambda_1)$ and diversity $(\lambda_2)$.
- **DiCE-Extended:** Uses a refined multi-objective optimization function with three weighted parameters: **Proximity $(\lambda_p)$**, **Diversity $(\lambda_d)$**, and **Robustness $(\lambda_r)$**. In experiments, $\lambda_r$ was typically set to 0.4 to ensure robustness contributes meaningfully without dominating the other objectives.

### 3. Performance and Validity

- **Validity:** DiCE-Extended consistently maintains a **100% validity rate** (meaning every suggestion successfully flips the model's outcome), even when generating a high number of counterfactuals ($k=10$) where standard DiCE might sometimes produce duplicates or invalid examples.
- **Proximity and Sparsity:** DiCE-Extended actually produces **closer** counterfactuals than the original. In benchmarks, the "median proximity" improved by 30% to 95% compared to standard DiCE. It also achieves better **sparsity**, meaning it suggests changing fewer features to reach the goal.
- **Execution Time:** DiCE-Extended is slightly more computationally demanding due to the extra robustness calculations. On average, it takes about **1.2x longer** to generate explanations than the original DiCE, but it is considered a worthy trade-off for the increased stability.

### Summary Table

|Feature|standard DiCE|DiCE-Extended|
|:--|:--|:--|
|**Primary Focus**|Diversity & Proximity|Robustness, Diversity & Proximity|
|**New Metric**|N/A|**Dice–Sørensen coefficient**|
|**Stability**|Sensitive to perturbations|Stable under minor input variations|
|**Complexity**|Fast but less robust|1.2x slower but higher quality|
|**Fidelity**|Local boundary approximation|Improved alignment with decision boundaries|

In short, **DiCE-Extended** is a more "reliable" version of the original. It takes the diversity logic of DiCE and hard-wires **resilience** into it, ensuring the explanations don't break when the data is slightly noisy or perturbed.