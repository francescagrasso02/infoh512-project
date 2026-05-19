# INFO-H512 Project — Recourse Fairness in Employee Attrition Prediction

## Research Question
*Are the actions suggested by an AI model to prevent employee attrition equally achievable for all demographic groups (e.g., gender, age), and does this fairness depend on the type of model used?*

## Overview
When an AI model predicts that an employee is likely to leave a company, it can also suggest actions the employee could take to change that prediction (e.g., change department, increase working hours). This is called **algorithmic recourse**.

This project investigates whether these suggested actions are equally easy to follow for all employees, regardless of gender or age — and whether the answer changes depending on which ML model is used.

## Methodology
1. **Dataset**: IBM HR Employee Attrition (publicly available on Kaggle)
2. **Models**: TBD — at least 2 models will be selected and compared
3. **Recourse generation**: DiCE (Diverse Counterfactual Explanations)
4. **Fairness evaluation**: FACTS framework — measuring recourse fairness across demographic subgroups using multiple notions (Burden, Equal Effectiveness, Equal Cost of Effectiveness, Equal Choice)

## Key References
- Mothilal et al. (2020) — DiCE: https://arxiv.org/abs/1905.07697
- Kavouras et al. (2023) — FACTS: https://arxiv.org/abs/2306.14978
- Karimi et al. (2022) — Algorithmic Recourse Survey: https://dl.acm.org/doi/10.1145/3527848
- Wachter et al. (2017) — Counterfactual Explanations: https://arxiv.org/abs/1711.00399

## Repository Structure
```
├── papers/         # Reference papers
├── src/            # Source code
├── data/           # Dataset
└── report/         # Final report (AAAI 2026 format)
```
