
# INFO-H512 — Project Master Plan

> Last updated: 2026-05-27
  
[[Our Topic]]
[[Questions Board]]
[[Proyecto - ES]]

---
## Research Question

> _Are the actions suggested by an AI model to prevent employee attrition equally achievable for all demographic groups (e.g., gender, age), and does this fairness depend on the type of ML model used?_


```cardlink
url: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data?select=WA_Fn-UseC_-HR-Employee-Attrition.csv
title: "IBM HR Analytics Employee Attrition & Performance"
description: "Predict attrition of your valuable employees"
host: www.kaggle.com
image: https://storage.googleapis.com/kaggle-datasets-images/1067/1925/5fc43e97e25975d1a14cba2e706255cb/dataset-card.jpg
```


---
## Pipeline (at a glance)
1. IBM HR Dataset 
2. Preprocessing
3. ML Models (≥2)
4. DiCE Counterfactuals
5. FACTS Fairness Analysis  
6. Results & Report

---
## Team & Responsibilities

| Phase                       | Owner               | Status         |
| --------------------------- | ------------------- | -------------- |
| Literature Review           | Francesca           | 🔄 In progress |
| Data Preprocessing & EDA    | Francesca           | ✅ Done         |
| ML Models                   | Francesca  & Kamila | ⏳ Pending      |
| DiCE Implementation         | Lisa                | ⏳ Pending      |
| FACTS Fairness Analysis     | Matteo              | ⏳ Pending      |
| Results & Visualization     | Matteo              | ⏳ Pending      |
| Final Report & Presentation | All                 | ⏳ Pending      |

---

## Phase-by-Phase Plan
### ✅ Phase 0 — Already done

- [x] GitHub repo set up
- [x] IBM HR Attrition dataset downloaded
- [x] EDA (class distribution, gender/age distribution)
- [x] Preprocessing (encoding, normalization, train/test split)

---

### 📚 Phase 1 — Literature (Francesca) — **URGENT**

Goal: everyone has enough context to implement their part.

| Paper                          | Action needed                      |
| ------------------------------ | ---------------------------------- |
| Mothilal et al. (2020) — DiCE  | Full read (Lisa + Francesca)       |
| Kavouras et al. (2023) — FACTS | Full read (Matteo)                 |
| Karimi et al. (2022) — Survey  | Skim abstract + key sections (All) |
| Wachter et al. (2017)          | Abstract + conclusion only (All)   |
| [[GLANCE.pdf]]                 | Kamila                             |

**Deadline suggestion:** before starting Phase 3 (DiCE).

---

### 🤖 Phase 2 — ML Models (Francesca) — #kamila 

**Decision needed first:**

- [x] Confirm model selection → ✅ 2026-05-28
	- [x] **Random Forest** #kamila ✅ 2026-05-28
	- [x] XGBoost** (as discussed in workflow) #kamila ✅ 2026-05-28
- [x] Decide evaluation metrics → accuracy + F1 (imbalanced classes → use F1 macro or weighted) ✅ 2026-05-28

**Tasks:**

- [x] Train Random Forest on preprocessed data ✅ 2026-05-28
- [x] Train XGBoost on preprocessed data ✅ 2026-05-28
- [x] Evaluate both (accuracy, F1, confusion matrix) ✅ 2026-05-28
- [x] Export trained models as `.pkl` or equivalent (needed by DiCE) ✅ 2026-05-28
- [x] Document which preprocessing pipeline is baked in (important for DiCE compatibility) ✅ 2026-05-28

**Key constraint:** DiCE needs the model wrapped in its API. Check DiCE docs for how to wrap sklearn/XGBoost models.

---

### ⚙️ Phase 3 — DiCE Counterfactual Generation (Lisa) — **Core work**

**Open decisions to resolve (block everything downstream):**

|#|Question|Proposed answer|To confirm|
|---|---|---|---|
|1|Which features are immutable?|`Age`, `Gender` (also `EmployeeNumber`, `Over18`)|✅ Agreed|
|2|How many CFs per employee?|3–4|Confirm with team|
|3|Which DiCE method?|`random` (fast) or `genetic` (better quality) — compare both|Decide|
|4|Proximity vs diversity weight|Test default (0.5 / 1.0) and one variant|Decide|
|5|Feature weights|Assign higher weight to hard-to-change features (e.g., `Education`, `JobLevel`)|Research needed|

**Tasks:**

- [ ] Install DiCE: `pip install dice-ml`
- [ ] Wrap trained models in DiCE's `Model` class
- [ ] Define `Data` object with feature types and immutable features
- [ ] Generate counterfactuals for all `Attrition=1` employees (test on subset first)
- [ ] Save CF output as structured DataFrame (employee ID + original features + CF features)
- [ ] Validate output quality (proximity, sparsity, validity)
- [ ] Run for **both** ML models (RF and XGBoost)

**Output format needed for FACTS:**

```
employee_id | group (gender/age) | model | cf_index | feature_1 | feature_2 | ... | prediction
```

---

### 📊 Phase 4 — FACTS Fairness Analysis (Matteo) — **Depends on Phase 3**

**FACTS metrics to compute:**

- [ ] **Burden** — how many features need to change on average per group
- [ ] **Equal Effectiveness** — same proportion of valid CFs across groups
- [ ] **Equal Cost of Effectiveness** — same effort needed for effective CFs
- [ ] **Equal Choice** — same number of diverse CFs available per group

**Tasks:**

- [ ] Study FACTS library / re-implement metrics from paper if no library available
- [ ] Apply metrics to DiCE output, segmented by: `Gender` (Male/Female) and `AgeGroup` (e.g., <35, 35–50, >50)
- [ ] Compare fairness metrics **across demographic groups** (for each model)
- [ ] Compare fairness metrics **across models** (RF vs XGBoost, for each group)
- [ ] Significance testing if time permits

**Key question for Matteo:** Does a FACTS Python library exist, or do we implement the metrics manually from the paper? → Resolve early.

---

### 📈 Phase 5 — Results & Visualization (Matteo + All)

- [ ] Summary table: fairness metrics by group × model
- [ ] Bar charts / heatmaps comparing groups
- [ ] Example CFs for 2–3 illustrative employees
- [ ] Critical analysis: which model is "fairer"? Why might one be worse?
- [ ] Limitations section (dataset bias, DiCE limitations, FACTS assumptions)

---

### 📝 Phase 6 — Report (AAAI 2026 format, max 4 pages)

**Structure:**

1. Introduction + Research Question
2. Related Work (DiCE, FACTS, algorithmic recourse)
3. Methodology (dataset, models, DiCE setup, FACTS metrics)
4. Experiments & Results
5. Discussion + Limitations
6. Conclusions
7. References

**Constraints:**

- Max **4 pages** (AAAI 2026 format)
- Use the provided report template in `Notes/Report Format/`
- Figures count toward page limit — plan space carefully

---

## 🚧 Immediate blockers to resolve (this week)

1. **Model selection confirmed?** → Random Forest + XGBoost? Anyone objecting?
2. **DiCE method?** → `random` vs `genetic` — Lisa should test both and decide
3. **FACTS library?** → Matteo checks if it exists as Python package or needs manual implementation
4. **Feature weights for DiCE** → Which IBM HR features are hardest to change in reality?
5. **Age grouping strategy** → How to bin age for fairness analysis? (<35 / 35-50 / >50 ?)

---

## Open Questions Log

|Question|Owner|Priority|
|---|---|---|
|Does a FACTS Python library exist?|Matteo|🔴 High|
|DiCE method: `random` vs `genetic`?|Lisa|🔴 High|
|Age bin thresholds for fairness groups|All|🟡 Medium|
|How to handle class imbalance in models?|Francesca|🟡 Medium|
|Feature weights for DiCE (effort-to-change)|Lisa + Francesca|🟡 Medium|
|Should we test a 3rd model (e.g., Logistic Regression as baseline)?|All|🟢 Low|

---

## Repository Structure (proposed)

```
infoh512-project/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb      ✅ done
│   ├── 02_models.ipynb                  ⏳ Francesca
│   ├── 03_dice_counterfactuals.ipynb   ⏳ Lisa
│   └── 04_facts_analysis.ipynb         ⏳ Matteo
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── dice_utils.py
│   └── facts_metrics.py
├── outputs/
│   ├── models/                          (saved .pkl models)
│   └── counterfactuals/                 (CF DataFrames per model)
├── Notes/                               (Obsidian notes)
└── report/
    └── infoh512_report.tex
```

---

## Timeline (suggested)

|Week|Goal|
|---|---|
|**Now**|Resolve all blockers in the table above|
|**Week +1**|Phase 2 done (models trained & exported) + Phase 1 papers read|
|**Week +2**|Phase 3 done (DiCE CFs generated for both models)|
|**Week +3**|Phase 4 done (FACTS metrics computed)|
|**Week +4**|Phase 5 + Phase 6 (results, visualizations, report draft)|
|**Final**|Report submitted|
