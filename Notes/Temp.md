#### Cell 5 — Custom threshold

python

```python
# ─── Custom Decision Threshold ────────────────────────────────────────────────
# The default threshold (0.5) is designed for balanced classes.
# With an imbalanced target, lowering the threshold increases sensitivity to
# attrition cases at the cost of more false positives.
#
# THRESHOLD = 0.2 was chosen by inspecting the F1-score for class 1 on the
# validation set across a range of thresholds, selecting the value that
# maximises recall without making precision collapse entirely.

THRESHOLD = 0.2

y_val_proba  = rf.predict_proba(X_val)[:, 1]
y_val_pred   = (y_val_proba  >= THRESHOLD).astype(int)

y_test_proba = rf.predict_proba(X_test)[:, 1]
y_test_pred  = (y_test_proba >= THRESHOLD).astype(int)
```

---

#### Cell 6 — Validation report

python

```python
# ─── Validation Set Report ────────────────────────────────────────────────────
# Evaluated on the held-out validation set to monitor performance during
# development without touching the test set.

print('✅ Predictions on validation set completed.')
print('\nClassification Report (Validation Set):')
print(classification_report(y_val, y_val_pred, target_names=['No (0)', 'Yes (1)']))
```

---

#### Cell 7 — Test report

python

```python
# ─── Test Set Report ──────────────────────────────────────────────────────────
# Final evaluation on the test set (never seen during training or threshold tuning).
# Given the class imbalance, F1 for class 1 (Attrition=Yes) is the primary metric:
#   - Accuracy alone is misleading (a model predicting all-No gets ~83%).
#   - Weighted F1 accounts for support but is dominated by the majority class.
#   - F1 for Attrition=1 directly measures the model's usefulness for HR use cases.

acc          = accuracy_score(y_test, y_test_pred)
f1_weighted  = f1_score(y_test, y_test_pred, average='weighted')
f1_attrition = f1_score(y_test, y_test_pred, pos_label=1)

print('=== TEST SET ===')
print(classification_report(y_test, y_test_pred, target_names=['No (0)', 'Yes (1)']))
print(f'Accuracy:          {acc:.3f}')
print(f'F1 (weighted):     {f1_weighted:.3f}')
print(f'F1 (Attrition=1):  {f1_attrition:.3f}  ← most important')
```

---

#### Cell 8 — Confusion matrix

python

```python
# ─── Confusion Matrix ─────────────────────────────────────────────────────────
# Visualises the trade-off between false positives (employees flagged as leaving
# who stay) and false negatives (employees who leave but go undetected).
# At threshold=0.2 we accept more false positives to reduce false negatives,
# consistent with the business cost of missing an at-risk employee.

fig, ax = plt.subplots(figsize=(5, 4))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred,
    display_labels=['No (stays)', 'Yes (leaves)'],
    cmap='Blues',
    ax=ax
)
ax.set_title(f'Confusion Matrix — Test Set (threshold={THRESHOLD})')
plt.tight_layout()
plt.show()
```

---

#### Cell 9 — Feature importances

python

```python
# ─── Feature Importances (MDI) ────────────────────────────────────────────────
# Mean Decrease in Impurity (MDI): measures how much each feature reduces Gini
# impurity on average across all trees. Higher = more discriminative.
# Note: MDI can overestimate the importance of high-cardinality features.
# The top features here serve as a qualitative guide for model interpretability
# and as inputs for the counterfactual generation step in DiCE.

importances = pd.Series(rf.feature_importances_, index=feature_names)
top15       = importances.nlargest(15)

fig, ax = plt.subplots(figsize=(8, 5))
top15.sort_values().plot(kind='barh', color='steelblue', ax=ax)
ax.set_title('Top 15 Feature Importances — Random Forest')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.show()
```

---

#### Cell 10 — Save artifacts

python

```python
# ─── Save Artifacts for DiCE ──────────────────────────────────────────────────
# Serialises all objects that DiCE (notebook 3) needs to reconstruct
# the exact preprocessing pipeline and generate valid counterfactuals:
#
#   random_forest.pkl  — trained classifier
#   scaler.pkl         — StandardScaler fitted on X_train
#   feature_names.pkl  — ordered list of feature names (column order matters)
#   label_encoders.pkl — LabelEncoders for each categorical feature

artifacts = {
    'random_forest.pkl':  rf,
    'feature_names.pkl':  feature_names,
    'label_encoders.pkl': label_encoders,
    'scaler.pkl':         scaler,
}

for filename, obj in artifacts.items():
    path = os.path.join(MODELS_PATH, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'✅ Saved: {path}')

print('\n📦 All artifacts saved.')
```