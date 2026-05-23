# Project Workflow and Task Assignment

## Goal

Investigate whether the actions suggested by an AI model to prevent employee attrition are equally achievable across demographic groups and whether this fairness changes depending on the ML model used.

Pipeline:

Dataset → ML Model → DiCE → Fairness Analysis (FACTS) → Results

---

| Project Phase | Responsible | Tasks |
|---------------|-------------|--------|
| Literature Review & Paper Collection | Francesca | Collect and organize papers, summarize key concepts, investigate useful resources |
| Data Preprocessing & EDA | Francesca  | Load dataset, clean data, encode categorical features, perform basic EDA, prepare train/test data |
| ML Models | Francesca | Implement Random Forest and XGBoost models, train and evaluate them |
| DiCE Methodology |  | Study DiCE library, test implementation, generate counterfactual explanations |
| Fairness / FACTS Analysis |  | Analyze whether suggested counterfactual actions differ across demographic groups |
| Results & Visualization |  | Generate plots, tables, and summarize findings |
| Final Report & Presentation |  | Write final report and prepare presentation |

---

## Notes

- Keep the implementation simple.
- Initial models: Random Forest and XGBoost.
- Focus mainly on the AI/DiCE and FACTS components.
- The main objective is not finding the best classifier, but studying the fairness of the suggested actions.
