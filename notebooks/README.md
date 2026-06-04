# INFO-H512 — Counterfactuals & Fairness on the IBM HR Attrition Dataset

**Course:** Current Trends in Artificial Intelligence (INFO-H512) · ULB
**Dataset:** IBM HR Employee Attrition (1470 employees, binary `Attrition` target, 16.1% positive)

This project predicts which employees are likely to leave the company, uses
**DiCE** to generate counterfactual explanations (personalised "what could you
change" action plans for at-risk employees), and then asks whether those
predictions and action plans are **equally fair across different groups of
employees**.

> All numbers in this README are taken from the saved notebook outputs of the
> current run. If you re-run the pipeline and results change, update this file.

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
> whole project reads as a single end-to-end sequence. Some notebook cells still
> reference the old numbers; the mapping is `04 → 3.a`, `04b → 3.b`, `05 → 4`,
> `06 → 5`.

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

Run the notebooks top to bottom in the order above. They **auto-detect** whether
you are on Google Colab or running locally (VSCode) and resolve all paths through
`get_project_paths()` in `utils.py` — no manual path editing needed.

- **Local (VSCode):** open a notebook from inside `notebooks/`; paths resolve
  relative to the project root.
- **Google Colab:** Drive is mounted automatically and the shared
  `Current_Trends_in_AI/` folder is located.

> ⚠️ **Notebooks 3.a and 3.b are the slow ones** — DiCE generates counterfactuals
> for every at-risk employee individually. Expect ~10 min for 3.a and ~30 min for
> 3.b (it re-runs DiCE three times). Notebooks 4 and 5 run in under a minute.

**Requirements:** all dependencies are listed in `requirements.txt`.

---

## `utils.py` — shared code

All logic shared across notebooks lives here so the notebooks stay clean and
don't duplicate code.

| Function / class | What it does |
|------------------|--------------|
| `get_project_paths()` | Auto-detects local vs Colab and returns the paths dict (`BASE`, `DATA`, `RF`, `XGBoost`, `FAIRNESS`) |
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
| `THRESHOLD_RF` | `0.489` | RF decision threshold (tuned on validation in 2.a) |
| `THRESHOLD_XGB` | `0.592` | XGBoost decision threshold (tuned on validation in 2.b) |
| `MAX_SALARY_INCREASE` | `0.50` | Default salary cap for recourse (+50%) |
| `AGE_THRESHOLD` | `35` | `AgeGroup = "Young"` if `Age < 35` else `"Senior"` |
| `PROTECTED_ATTRIBUTES` | `["Gender", "AgeGroup", "MaritalStatus"]` | Attributes audited for fairness |
| `IMMUTABLE` | `DEMOGRAPHIC + TIME + HISTORICAL` | Features DiCE may not change |
| `PERMITTED_RANGE` | dict | Allowed ranges for mutable ordinal features |

`IMMUTABLE` is built from three groups:
- **Demographic:** Age, Gender, MaritalStatus, EducationField, Department
- **Time:** YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, TotalWorkingYears, NumCompaniesWorked
- **Historical:** PercentSalaryHike, PerformanceRating, Education, HourlyRate, DailyRate, MonthlyRate

> ⚠️ `THRESHOLD_RF` and `THRESHOLD_XGB` must be updated if the optimal thresholds
> change after a re-run of the model notebooks.

---

## Notebook 1 — Data Preprocessing

Loads the raw CSV, runs exploratory data analysis (shape, types, missing values,
class distribution), cleans the data, and saves scaled train/val/test splits plus
the fitted scaler, feature names, and label encoders to `data/processed/`. Splits
are 70/15/15 stratified, seed 42.

It also defines the **DiCE feature taxonomy** — classifying every feature as
immutable / ordinal / nominal / continuous — the prerequisite for any
counterfactual configuration downstream. Constant columns (`EmployeeCount`,
`StandardHours`, `EmployeeNumber`, `Over18`) are dropped because they carry no
signal.

---

## Notebooks 2.a & 2.b — Random Forest and XGBoost

Each notebook trains one classifier on the preprocessed splits, runs a
hyperparameter search (RandomizedSearchCV, 5-fold stratified, optimising recall on
the attrition class), tunes an optimal decision threshold on the validation set,
evaluates on the test set (ROC / PR curves, confusion matrix, MDI feature
importances), and exports the model with its scaler, label encoders, and feature
names so DiCE can later reconstruct the exact same pipeline.

**Test-set results (n = 221):**

| Metric | Random Forest | XGBoost |
|--------|---------------|---------|
| Accuracy | 0.851 | 0.805 |
| F1 (Attrition = 1) | 0.459 | 0.506 |
| ROC-AUC | 0.996 | 0.940 |
| PR-AUC | 0.978 | 0.797 |
| Threshold τ | 0.489 | 0.592 |
| Recall (Attrition) | 90.7% | 83.1% |
| Precision (Attrition) | 86.3% | 60.6% |

**At-risk population entering the recourse analysis (full dataset, n = 1470):**

| Model | At-risk flagged | Recall | Precision |
|-------|-----------------|--------|-----------|
| Random Forest | 249 (16.9%) | 90.7% | 86.3% |
| XGBoost | 325 (22.1%) | 83.1% | 60.6% |

XGBoost is more sensitive on the minority class (higher F1) but much less precise:
~40% of its flagged employees are false positives, so it carries a larger and
noisier at-risk population into the fairness analysis. Random Forest has stronger
overall discrimination (ROC-AUC / PR-AUC).

---

## Notebook 3.a — Recourse Dataset + Level 1: Prediction Fairness

1. Loads the RF and XGBoost models.
2. Runs DiCE on **all** at-risk employees, producing a recourse dataset with one
   row per employee: whether a counterfactual was found, its L1 cost, and which
   features changed.
3. Computes **Level 1** prediction fairness — statistical parity, equal
   opportunity, predictive equality — across Gender, AgeGroup, and MaritalStatus.

**Recourse coverage (+50% cap):**

| Model | At-risk processed | Recourse found | Mean L1 | Median L1 |
|-------|-------------------|----------------|---------|-----------|
| Random Forest | 249 | 243 (97.6%) | 0.992 | 0.872 |
| XGBoost | 325 | 297 (91.4%) | 1.206 | 0.911 |

**Level 1 prediction-fairness gaps (max − min across groups, Random Forest):**

| Attribute | Pred. rate gap | Recall gap | FPR gap |
|-----------|----------------|------------|---------|
| Gender | 0.010 | 0.020 | 0.007 |
| AgeGroup | 0.175 | 0.125 | 0.058 |
| MaritalStatus | 0.184 | 0.069 | 0.050 |

Gender is balanced. The large AgeGroup and MaritalStatus gaps (Young flagged at
26.7% vs 9.2% Senior; Single at 27.9% vs 9.5% Divorced) reflect genuine attrition
patterns in the data, not a model artefact — but they motivate the deeper analysis
at Levels 2 and 3.

---

## Notebook 3.b — Sensitivity Analysis: Salary Cap

Re-runs the full DiCE pipeline three times under different salary caps to test
whether the high effectiveness in 3.a is robust or just an artefact of a generous
budget.

| Config | Cap | Rationale |
|--------|-----|-----------|
| `tight` | +20% | Realistic retention offer |
| `moderate` | +35% | Generous but plausible |
| `loose` | +50% | Baseline (matches 3.a) |

Kept separate from 3.a deliberately: 3.a already takes ~10 min, so folding the
sweep into it would make every re-run take 40+ min.

**Equal Effectiveness gap vs salary cap (XGBoost worsens; RF stays stable):**

| Attribute | Model | loose (+50%) | moderate (+35%) | tight (+20%) |
|-----------|-------|-------------|-----------------|--------------|
| AgeGroup | RF | 0.034 | 0.034 | 0.040 |
| AgeGroup | XGB | 0.123 | 0.136 | 0.145 |
| MaritalStatus | RF | 0.046 | 0.046 | 0.053 |
| MaritalStatus | XGB | 0.119 | 0.136 | 0.136 |

---

## Notebook 4 — Level 2: Counterfactual Fairness (Kusner et al., 2017)

Individual-level test: for each employee, flip Gender (Male ↔ Female) or shift Age
15 years across the threshold of 35, holding all else constant, and record whether
the prediction changes. A non-zero flip rate means the model depends on that
attribute. Run on the full population and on at-risk employees only.

**MaritalStatus is excluded:** with three categories there is no natural binary
flip — choosing a target would make the result depend on the chosen direction.

| Population | Model | Gender flip | Age shift |
|------------|-------|-------------|-----------|
| Full (n = 1470) | Random Forest | 1.9% | 13.4% |
| Full (n = 1470) | XGBoost | 2.0% | 6.8% |
| At-risk only | Random Forest (n = 249) | 0.0% | 0.0% |
| At-risk only | XGBoost (n = 325) | 0.0% | 0.0% |

**Key result:** Gender is essentially fair for both models. Age shows moderate
dependence across the full population (more so for RF). But among employees already
flagged as at-risk, the flip rate is **0% for both attributes and both models** —
once an employee is flagged, changing their gender or age does not change the
verdict. The model uses protected attributes only as a general population-level
risk correlate, not to decide *who gets flagged*.

---

## Notebook 5 — Level 3: FACTS, Fairness of the Recourse (Kavouras et al., NeurIPS 2023)

Computes the four FACTS notions on the recourse datasets from 3.a (baseline) and
3.b (sensitivity). The cheap-recourse budget is the **global median L1 cost = 0.889**.

| Notion | Question | Measure |
|--------|----------|---------|
| Equal Effectiveness (EE) | Same share of each group finds recourse? | share with ≥1 valid CF |
| Cost of Effectiveness (CoE) | Same cost to find recourse? | median L1 cost |
| Effectiveness Within Budget (EWB) | Same share finds *cheap* recourse? | share with L1 ≤ 0.889 |
| Choice for Recourse | Same number of options? | mean number of CFs |

The **unfairness score** for each notion is the max − min gap across subgroups
(0 = perfectly fair). Gaps above 0.05 are flagged as material.

**Baseline FACTS unfairness gaps (+50% cap):**

| Model | Attribute | EE | CoE | EWB | Choice |
|-------|-----------|------|------|------|--------|
| RF | Gender | 0.023 | 0.067 | 0.063 | 0.00 |
| RF | AgeGroup | 0.034 | 0.018 | 0.022 | 0.00 |
| RF | MaritalStatus | 0.046 | 0.077 | **0.172** | 0.00 |
| XGB | Gender | 0.026 | 0.002 | 0.040 | 0.02 |
| XGB | AgeGroup | **0.123** | **0.242** | **0.274** | 0.02 |
| XGB | MaritalStatus | **0.119** | **0.119** | **0.185** | 0.02 |

**Key result:** Equal Effectiveness gaps at baseline are near zero — so a shallow
audit would call both models fair. But Cost and within-Budget gaps tell a different
story. **Random Forest** keeps every gap below threshold except one (MaritalStatus
EWB = 0.172, worth monitoring). **XGBoost fails materially on AgeGroup and
MaritalStatus** across EE, CoE, and EWB — and (per 3.b) those gaps widen as the
salary budget tightens.

---

## Main conclusion

The models look fair on prediction rates (Level 1) and at the individual level
(Level 2 — among flagged employees the protected-attribute flip rate is 0%). But
the recourse analysis (Level 3) shows that some subgroups — most clearly
**Young and Single employees under XGBoost** — face systematically harder or more
expensive paths to the same outcome. This disparity is invisible to Equal
Effectiveness and only surfaces through Cost-of-Effectiveness and
Effectiveness-within-Budget.

**XGBoost distributes recourse cost more unequally than Random Forest**, and that
inequality grows as the employer's budget shrinks. If the goal is to retain
employees *equitably*, Random Forest is the fairer choice — not because it predicts
better, but because its recourse is more evenly distributed. This empirically
supports the claim of Von Kügelgen et al. (2022) that **recourse fairness is
complementary to prediction fairness**.

---

## References

- Kavouras et al. (2023). *FACTS: Fairness-Aware Counterfactuals for Subgroups.* NeurIPS 2023.
- Kusner et al. (2017). *Counterfactual Fairness.* NeurIPS 2017.
- Von Kügelgen et al. (2022). *On the Fairness of Causal Algorithmic Recourse.* AAAI 2022.
- Mothilal et al. (2020). *Explaining ML Classifiers through Diverse Counterfactual Explanations (DiCE).* ACM FAT*.
- Wachter et al. (2017). *Counterfactual Explanations without Opening the Black Box.* Harvard JOLT.
