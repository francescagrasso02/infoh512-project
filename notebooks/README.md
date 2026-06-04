# INFO-H512 — Counterfactuals & Fairness on the IBM HR Attrition Dataset

**Course:** Current Trends in Artificial Intelligence (INFO-H512) · ULB
**Dataset:** IBM HR Employee Attrition (1470 employees, binary `Attrition` target)

This project predicts which employees are likely to leave the company, uses
**DiCE** to generate counterfactual explanations (personalised "what could you
change" action plans for at-risk employees), and then asks whether those
predictions and action plans are **equally fair across different groups of
employees**.

---

## Pipeline overview

The notebooks run **in order**. Each one depends on the outputs of the previous.

```
1  →  2.a / 2.b  →  3.a  →  3.b  →  4  →  5
```

| # | Notebook | What it produces | Author |
|---|----------|------------------|--------|
| 1 | `1_Preprocessing.ipynb` | Cleaned, scaled train/val/test splits + DiCE feature taxonomy | Team |
| 2.a | `2_a__Random_Forest.ipynb` | Trained Random Forest + saved artifacts | Kamila |
| 2.b | `2_b__XGboost.ipynb` | Trained XGBoost + saved artifacts | Kamila |
| 3.a | `3_a__Recourse & Prediction Fairness.ipynb` | Recourse dataset (DiCE on all at-risk employees) + **Level 1** fairness | Matteo |
| 3.b | `3_b__Sensitivity_analysis.ipynb` | Recourse datasets under three salary-cap configs | Matteo |
| 4 | `4__Counterfactual_Fairness.ipynb` | **Level 2** individual counterfactual fairness | Matteo |
| 5 | `5__FACTS.ipynb` | **Level 3** FACTS recourse fairness + final synthesis | Matteo |

> **Note on numbering.** Earlier drafts of the fairness notebooks were numbered
> `04 → 04b → 05 → 06`. They have been renumbered to `3.a → 3.b → 4 → 5` so the
> whole project reads as a single end-to-end sequence. If you find a stray
> reference to "notebook 04/05/06" inside a notebook cell, it maps as:
> `04 → 3.a`, `04b → 3.b`, `05 → 4`, `06 → 5`.

---

## The three levels of fairness

Each level asks a deeper question than the one before it.

| Level | Question | Notebook |
|-------|----------|----------|
| 1 — Prediction | Does the model flag attrition at the same rate for all groups? | `3.a` |
| 2 — Counterfactual | Would the model change its mind if a protected attribute were different? | `4` |
| 3 — Recourse (FACTS) | Is the recourse DiCE recommends equally achievable for all groups? | `5` |

Protected attributes analysed: **Gender**, **AgeGroup** (Young < 35 ≤ Senior),
and **MaritalStatus**.

---

## Repository layout

```
infoh512-project/
├── notebooks/
│   ├── utils.py                                   ← shared functions & constants
│   ├── 1_Preprocessing.ipynb
│   ├── 2_a__Random_Forest.ipynb
│   ├── 2_b__XGboost.ipynb
│   ├── 3_a__Recourse & Prediction Fairness.ipynb
│   ├── 3_b__Sensitivity_analysis.ipynb
│   ├── 4__Counterfactual_Fairness.ipynb
│   └── 5__FACTS.ipynb
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv      ← raw dataset
│   ├── processed/                                 ← splits + scaler + encoders (notebook 1)
│   │   ├── X_train.npy / X_val.npy / X_test.npy
│   │   ├── y_train.npy / y_val.npy / y_test.npy
│   │   ├── scaler.pkl
│   │   ├── feature_names.pkl
│   │   └── label_encoders.pkl
│   ├── models/
│   │   ├── RF/        ← random_forest.pkl + scaler/encoders/feature_names (notebook 2.a)
│   │   └── XGBoost/   ← xgboost.pkl + scaler/encoders/feature_names (notebook 2.b)
│   └── fairness/                                  ← all fairness outputs (created automatically)
│       ├── recourse_RF.csv
│       ├── recourse_XGB.csv
│       ├── level1_prediction_fairness.png
│       ├── level2_cf_fairness_full.csv
│       ├── level2_cf_fairness_atrisk.csv
│       ├── level2_counterfactual_fairness.png
│       ├── level3_facts_gaps.csv
│       ├── level3_effectiveness.png
│       ├── level3_cost.png
│       ├── level3_feature_changes_gender.png
│       ├── sensitivity_summary.csv
│       ├── sensitivity_facts_gaps.csv
│       └── sensitivity_[tight|moderate|loose]_[RF|XGB].csv
└── requirements.txt
```

---

## How to run

Run the notebooks top to bottom in the order above. The notebooks **auto-detect**
whether you are on Google Colab or running locally (VSCode) and set their paths
through `get_project_paths()` in `utils.py` — no manual path editing needed.

- **Local (VSCode):** open a notebook from inside `notebooks/`; paths resolve
  relative to the project root.
- **Google Colab:** Drive is mounted automatically and the shared
  `Current_Trends_in_AI/` folder is located.

> ⚠️ **Notebooks 3.a and 3.b are the slow ones** — DiCE generates counterfactuals
> for every at-risk employee (~280 people). Expect ~10 min for 3.a and ~30 min
> for 3.b (it re-runs DiCE three times). Notebooks 4 and 5 run in under a minute.

**Requirements:** all dependencies are listed in `requirements.txt`.

---

## `utils.py` — shared code

All logic shared across notebooks lives here so the notebooks stay clean and
don't duplicate code.

| Function / class | What it does |
|------------------|--------------|
| `get_project_paths()` | Auto-detects local vs Colab and returns the project paths dict (`BASE`, `DATA`, `RF`, `XGBoost`, `FAIRNESS`) |
| `load_artifacts(models_dir)` | Loads model + scaler + label encoders + feature names from a model directory |
| `load_raw_with_split()` | Reloads the raw CSV and rebuilds the label-encoded, unscaled frame (recovers readable demographics + `AgeGroup`) |
| `load_pickle(path)` | Small pickle-loading helper |
| `ScaledModelWrapper` | Wraps a model + its scaler so DiCE can pass **unscaled** data; applies the scaler internally on `predict` / `predict_proba` |
| `build_recourse_dataset()` | Runs DiCE on all at-risk employees and returns a tidy DataFrame (one row per employee) |
| `decode_row()` | Converts integer-encoded categoricals back to readable labels |
| `mean_l1()` | Normalised L1 distance between a query and its counterfactuals |
| `changed_features()` | Which features differ between a query row and a counterfactual |

### Key constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `SEED_PREPROCESSING` | `42` | Split seed used in notebook 1 |
| `SEED_DICE` | `48` | Seed used by the model / DiCE notebooks |
| `THRESHOLD_RF` | `0.489` | RF decision threshold (re-tuned in 2.a) |
| `THRESHOLD_XGB` | `0.592` | XGBoost decision threshold (re-tuned in 2.b) |
| `MAX_SALARY_INCREASE` | `0.50` | Default salary cap for recourse (+50%) |
| `AGE_THRESHOLD` | `35` | `AgeGroup = "Young"` if `Age < 35` else `"Senior"` |
| `PROTECTED_ATTRIBUTES` | `["Gender", "AgeGroup", "MaritalStatus"]` | Attributes audited for fairness |
| `IMMUTABLE` | `DEMOGRAPHIC + TIME + HISTORICAL` | Features DiCE may not change |
| `PERMITTED_RANGE` | dict | Allowed ranges for mutable ordinal features |

`IMMUTABLE` is built from three groups:
- **Demographic:** Age, Gender, MaritalStatus, EducationField, Department
- **Time:** YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, TotalWorkingYears, NumCompaniesWorked
- **Historical:** PercentSalaryHike, PerformanceRating, Education, HourlyRate, DailyRate, MonthlyRate

> ⚠️ `THRESHOLD_RF` and `THRESHOLD_XGB` are placeholders that **must be updated**
> after (re)running the model notebooks if the optimal thresholds change.

---

## Notebook 1 — Data Preprocessing

Loads the raw CSV, runs exploratory data analysis (shape, types, missing values,
class distribution), cleans the data, and saves scaled train/val/test splits plus
the fitted scaler, feature names, and label encoders to `data/processed/`.

It also defines the **DiCE feature taxonomy** — classifying every feature as
immutable / ordinal / nominal / continuous — which is the prerequisite for any
counterfactual configuration downstream. The constant `Over18` column is dropped
because it carries no information.

---

## Notebooks 2.a & 2.b — Random Forest and XGBoost

Each notebook trains one classifier on the preprocessed splits, runs a
hyperparameter search, tunes an optimal decision threshold, evaluates on the test
set (ROC / PR curves, confusion matrix, MDI feature importances), and exports the
model together with its scaler, label encoders, and feature names so DiCE can
later reconstruct the exact same pipeline.

**Headline results:**
- **Random Forest** flags 281 at-risk employees — recall 88.2%, precision 74.4% (more cautious).
- **XGBoost** flags 227 — recall 81.9%, precision 85.5% (more precise, less cautious).

---

## Notebook 3.a — Recourse Dataset + Level 1: Prediction Fairness

1. Loads the RF and XGBoost models.
2. Runs DiCE on **all** at-risk employees (not just a small sample), producing a
   recourse dataset with one row per employee: whether a counterfactual was found,
   its L1 cost, and which features changed.
3. Computes **Level 1** prediction fairness — statistical parity, equal
   opportunity, predictive equality — across Gender, AgeGroup, and MaritalStatus.

**Key result:** both models find recourse for ~99–100% of at-risk employees at the
+50% cap, and group-level prediction gaps are small. This is where the analysis
*starts* — the interesting disparities only appear at Levels 2 and 3.

---

## Notebook 3.b — Sensitivity Analysis: Salary Cap

Re-runs the full DiCE pipeline three times under different salary-cap constraints
to test whether the near-100% effectiveness in 3.a is robust or just an artefact
of a generous budget.

| Config | Cap | Rationale |
|--------|-----|-----------|
| `tight` | +20% | Realistic retention offer |
| `moderate` | +35% | Generous but plausible |
| `loose` | +50% | Original baseline (matches 3.a) |

Kept separate from 3.a deliberately: 3.a takes ~10 min, so folding the sensitivity
sweep into it would make every re-run take 40+ min. Same reproducibility logic as
separating preprocessing from analysis.

**Key result:** effectiveness stays high (~97–100%) at all caps, but **XGBoost's
gaps grow as the cap tightens** (e.g. MaritalStatus effectiveness gap 0.017 → 0.034;
AgeGroup 0.013 → 0.026), while **Random Forest stays stable** near zero.

---

## Notebook 4 — Level 2: Counterfactual Fairness (Kusner et al., 2017)

Individual-level test: for each employee, flip a protected attribute (Gender or
Age) and check whether the prediction changes. A high flip rate means the model
depends on that attribute. Run on the full population (1470) and on at-risk
employees only.

**MaritalStatus is excluded** here: with three categories, flipping requires
choosing an arbitrary target, which would make the flip rate depend on the chosen
direction rather than on the model's fairness. Gender and Age have natural,
unambiguous flips.

| Model | Flip Gender (full) | Flip Age (full) | Flip Age (at-risk) |
|-------|--------------------|-----------------|--------------------|
| Random Forest | 0.7% | 6.8% | 10.7% |
| XGBoost | 0.5% | 4.9% | 10.6% |

**Key result:** Gender is essentially fair (<1% flips) for both models; Age shows
moderate dependence that **roughly doubles among at-risk employees** — Age weighs
more heavily on *who gets flagged* than on predictions in general.

---

## Notebook 5 — Level 3: FACTS, Fairness of the Recourse (Kavouras et al., NeurIPS 2023)

Computes the four FACTS notions on the recourse datasets from 3.a (baseline) and
3.b (sensitivity).

| Notion | Question | Measure |
|--------|----------|---------|
| Equal Effectiveness | Same share of each group finds recourse? | share with ≥1 valid CF |
| Equal Cost of Effectiveness | Same cost to find recourse? | median L1 cost |
| Equal Effectiveness within Budget | Same share finds *cheap* recourse? | share with L1 ≤ global median |
| Equal Choice for Recourse | Same number of options? | mean number of CFs |

The **unfairness score** for each notion is the max−min gap across subgroups (0 =
perfectly fair).

**Key result — baseline (+50% cap):** Equal Effectiveness gaps are near zero (so a
shallow audit would call the models fair), but **Cost** and **within-Budget** gaps
reveal that **Single employees pay systematically more** for the same recourse
(e.g. 62% of Single employees must change StockOptionLevel vs 19% of Divorced).
**XGBoost is consistently more unfair than RF** across attributes, and the gap
widens as the salary budget tightens.

---

## Main conclusion

The models look fair on prediction rates (Level 1) and are largely fair at the
individual level for Gender (Level 2). But the recourse analysis (Level 3) shows
that **Single employees — and to a lesser extent some subgroups — face systematically
higher costs to reach the same outcome**. This disparity is invisible to Equal
Effectiveness and only surfaces through the Cost-of-Effectiveness and
Effectiveness-within-Budget measures.

**XGBoost distributes recourse cost more unequally than Random Forest**, and that
inequality grows as the employer's budget shrinks. If the goal is to retain
employees *equitably*, Random Forest is the fairer choice — not because it predicts
better, but because its recourse is more evenly distributed. This empirically
demonstrates the claim of Von Kügelgen et al. (2022) that **recourse fairness is
complementary to prediction fairness**.

---

## References

- Kavouras et al. (2023). *FACTS: Fairness-Aware Counterfactuals for Subgroups.* NeurIPS 2023.
- Kusner et al. (2017). *Counterfactual Fairness.* NeurIPS 2017.
- Von Kügelgen et al. (2022). *On the Fairness of Causal Algorithmic Recourse.* AAAI 2022.
- Mothilal et al. (2020). *Explaining ML Classifiers through Diverse Counterfactual Explanations (DiCE).* ACM FAT*.
- Wachter et al. (2017). *Counterfactual Explanations without Opening the Black Box.* Harvard JOLT.
