# Notebooks

## 01 — Data Preprocessing

### What this notebook does

1. Loads the HR Attrition dataset directly from GitHub
2. Explores the data (shape, types, missing values, class distribution)
3. Cleans and preprocesses the features
4. Saves the train/val/test splits ready for modelling

### Output

The notebook saves the following files to your Google Drive (`Current_Trends_in_AI/preprocessed/`):

| File | Description |
|---|---|
| `X_train.npy` | Training features (scaled) |
| `X_val.npy` | Validation features (scaled) |
| `X_test.npy` | Test features (scaled) |
| `y_train.npy` | Training labels |
| `y_val.npy` | Validation labels |
| `y_test.npy` | Test labels |

> ⚠️ The first cell will ask you to authenticate with Google Drive — this is needed to save the preprocessed files. If you prefer not to connect Drive, the notebook still runs fine and you can download the arrays manually.

# Fairness Analysis — IBM HR Attrition Dataset
**Author:** Matteo | Course: INFO-H512 — Current Trends in AI | ULB

---

## What this part of the project does

The team trained two models (Random Forest and XGBoost) to predict which employees are likely to leave the company, and used DiCE to generate counterfactual explanations — personalised action plans that tell each at-risk employee what they could change to reduce their risk.

This part of the project asks a different question: **are those predictions and action plans equally fair across different groups of employees?**

We analyse fairness at three levels, each one deeper than the previous:

| Level | Question | Notebook |
|-------|----------|----------|
| 1 | Does the model predict attrition at the same rate for all groups? | `04` |
| 2 | Would the model change its mind if an employee's protected attribute were different? | `05` |
| 3 | Is the recourse — the set of changes DiCE recommends — equally achievable for all groups? | `06` |

---

## Files

```
notebooks/
├── utils.py                              ← shared functions used by all notebooks
├── 04_recourse_and_prediction_fairness.ipynb
├── 04b_sensitivity_analysis.ipynb
├── 05_counterfactual_fairness.ipynb
└── 06_FACTS.ipynb

data/fairness/                            ← all outputs (generated automatically)
├── recourse_RF.csv                       ← recourse dataset, Random Forest
├── recourse_XGB.csv                      ← recourse dataset, XGBoost
├── level1_prediction_fairness.png
├── level2_cf_fairness_full.csv
├── level2_cf_fairness_atrisk.csv
├── level2_counterfactual_fairness.png
├── level3_facts_gaps.csv
├── level3_effectiveness.png
├── level3_cost.png
├── level3_feature_changes_gender.png
├── sensitivity_summary.csv
├── sensitivity_facts_gaps.csv
└── sensitivity_[tight|moderate|loose]_[RF|XGB].csv
```

---

## How to run

Run the notebooks **in order**. Each one depends on the outputs of the previous.

```
04  →  04b  →  05  →  06
```

> ⚠️ Notebooks 04 and 04b are the slow ones — DiCE generates counterfactuals
> for every at-risk employee (~280 people). Expect 10 minutes for 04 and
> 30 minutes for 04b. Notebooks 05 and 06 run in under a minute.

**Requirements:** all dependencies are in the project `requirements.txt`.
The notebooks auto-detect whether you are running locally (VSCode) or on
Google Colab and set the paths accordingly.

---

## `utils.py` — shared functions

All shared code lives here so the notebooks stay clean and don't duplicate logic.

| Function | What it does |
|----------|-------------|
| `get_project_paths()` | Auto-detects local vs Colab and returns all project paths |
| `load_artifacts(dir)` | Loads model + scaler + encoders from a model directory |
| `load_raw_with_split()` | Downloads the CSV and rebuilds the label-encoded dataset |
| `ScaledModelWrapper` | Wraps a model + scaler so DiCE can work with unscaled data |
| `build_recourse_dataset()` | Runs DiCE on all at-risk employees and returns a tidy DataFrame |
| `mean_l1()` | Computes the normalised L1 distance between a query and its CFs |
| `changed_features()` | Returns which features differ between a query row and a CF |

**Key constants** (copied from the team's DiCE notebook to ensure consistency):
- `IMMUTABLE` — features DiCE cannot change (Age, Gender, Department, etc.)
- `PERMITTED_RANGE` — allowed ranges for each mutable feature
- `PROTECTED_ATTRIBUTES = ["Gender", "AgeGroup", "MaritalStatus"]`
- `THRESHOLD = 0.20` — probability threshold used by all models

---

## Notebook 04 — Recourse Dataset + Prediction Fairness

**What it does:**
1. Loads the team's RF and XGBoost models
2. Runs DiCE on **all** employees predicted as at-risk (not just the 3-person sample in the team's DiCE notebook)
3. Saves a recourse dataset with one row per at-risk employee: whether they found a counterfactual, how much it costs (L1), which features changed
4. Computes Level 1 fairness: statistical parity, equal opportunity, and predictive equality across Gender, AgeGroup, and MaritalStatus

**Key results:**
- Random Forest flags 281 at-risk employees (recall 88.2%, precision 74.4%)
- XGBoost flags 227 (recall 81.9%, precision 85.5%) — more precise, less cautious
- Both models find recourse for ~99-100% of at-risk employees at the +50% salary cap
- Prediction fairness gaps are small at group level — this is where the analysis starts

---

## Notebook 04b — Sensitivity Analysis: Salary Cap

**Why it exists separately from 04:**
Notebook 04 takes ~10 minutes to run. If we put the sensitivity analysis in the same notebook, every re-run would take 40+ minutes. Separating them means 04 runs once and produces the baseline; 04b explores variations independently.

**What it does:**
Re-runs the full DiCE pipeline three times with different salary cap constraints:

| Config | Cap | Rationale |
|--------|-----|-----------|
| `tight` | +20% | Realistic retention offer |
| `moderate` | +35% | Generous but plausible |
| `loose` | +50% | Original team configuration (baseline) |

**Key results:**
- Effectiveness stays high (~97-100%) at all salary cap levels — restringere il budget non crea disparità drammatiche
- However, **XGBoost shows growing gaps** as the cap tightens:
  - MaritalStatus effectiveness gap: 0.017 (loose) → 0.034 (tight)
  - AgeGroup effectiveness gap: 0.013 (loose) → 0.026 (tight)
- Random Forest remains stable across all configurations — gaps stay near zero

---

## Notebook 05 — Counterfactual Fairness

**What it does:**
Implements the individual-level fairness test from Kusner et al. (2017): for each employee, flip a protected attribute (Gender or Age) and check whether the model's prediction changes. A high flip rate means the model depends on that attribute.

The test runs on two populations:
- **Full population (1470 employees):** does the model treat anyone differently based on their protected attributes?
- **At-risk employees only:** among those already flagged, does flipping an attribute save them?

**Why MaritalStatus is excluded:**
MaritalStatus has three categories (Single/Married/Divorced). Flipping it requires choosing a target category, which introduces an arbitrary choice that would make the flip rate depend on the direction chosen rather than on the model's fairness. Gender and Age have natural, unambiguous flip operations.

**Key results:**

| Model | Flip Gender (full) | Flip Age (full) | Flip Age (at-risk) |
|-------|-------------------|-----------------|-------------------|
| Random Forest | 0.7% | 6.8% | 10.7% |
| XGBoost | 0.5% | 4.9% | 10.6% |

- **Gender: both models are fair** — flipping Gender changes less than 1% of predictions
- **Age: moderate dependence** — 6-7% of all employees change outcome; this rises to ~10.7% among at-risk employees, meaning Age plays a stronger role in deciding *who gets flagged* than in general predictions
- The at-risk flip rate being ~double the full-population rate is the key finding: Age influences the flagging decision disproportionately

---

## Notebook 06 — FACTS: Fairness of the Recourse

**What it does:**
Implements the four fairness notions from FACTS (Kavouras, Sacharidis et al., NeurIPS 2023) on the recourse datasets produced by notebooks 04 and 04b.

**The four FACTS notions:**

| Notion | Question | How we measure it |
|--------|----------|-------------------|
| Equal Effectiveness | Does the same share of each group find recourse? | share with ≥1 valid CF |
| Equal Cost of Effectiveness | Does each group pay the same cost? | median L1 cost |
| Equal Effectiveness within Budget | Does the same share find *cheap* recourse? | share with L1 ≤ global median cost |
| Equal Choice for Recourse | Does each group get the same number of options? | mean number of CFs |

**Key results — baseline (+50% salary cap):**

The most important finding is that **Equal Effectiveness gaps are near zero** — all groups find recourse at similar rates. This would suggest the models are fair. However:

| Model | Attribute | Cost gap | Within-budget gap |
|-------|-----------|----------|------------------|
| RF | MaritalStatus | 0.125 ⚠️ | 0.195 ⚠️ |
| XGBoost | MaritalStatus | 0.159 ⚠️ | 0.242 ⚠️ |
| RF | Gender | 0.052 | 0.070 |
| XGBoost | Gender | 0.043 | 0.005 |

**Single employees pay systematically more** for the same recourse:
- 62% of Single employees must change their StockOptionLevel vs 19% of Divorced
- 48% must increase training vs 27% of Married employees

**XGBoost is consistently more unfair than RF** across all attributes and metrics — and this gap widens as the salary cap tightens (sensitivity analysis).

---

## Main conclusion

> The models appear fair when looking at prediction rates (Level 1) and are largely
> fair at the individual level for Gender (Level 2). However, the recourse analysis
> (Level 3) reveals that **Single employees and, to a lesser extent, Male employees
> face systematically higher costs to achieve the same outcome**. This disparity
> is hidden by Equal Effectiveness metrics and only surfaces through the
> Cost of Effectiveness and Effectiveness within Budget measures from FACTS.
>
> Furthermore, **XGBoost distributes recourse costs more unequally than Random
> Forest**, and this inequality grows as the employer's salary budget decreases.
> If the goal is to retain employees equitably, Random Forest is the fairer
> choice — not because it predicts better, but because its recourse is more
> equally distributed.

This finding directly operationalises the claim of Von Kügelgen et al. (2022)
that *recourse fairness is complementary to prediction fairness* — and
demonstrates it empirically on a real HR dataset with the team's own models.

---

## References

- Kavouras et al. (2023). *FACTS: Fairness-Aware Counterfactuals for Subgroups*. NeurIPS 2023.
- Kusner et al. (2017). *Counterfactual Fairness*. NeurIPS 2017.
- Von Kügelgen et al. (2022). *On the Fairness of Causal Algorithmic Recourse*. AAAI 2022.
- Wachter et al. (2017). *Counterfactual Explanations without Opening the Black Box*. Harvard JOLT.