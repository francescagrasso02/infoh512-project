# Notebooks

## 01 — Data Preprocessing

### What this notebook does

1. Loads the HR Attrition dataset directly from GitHub
2. Explores the data (shape, types, missing values, class distribution)
3. Cleans and preprocesses the features
4. Saves the train/val/test splits ready for modelling

### Output

The notebook saves the following files to your Google Drive (`HR_Attrition_Project/preprocessed/`):

| File | Description |
|---|---|
| `X_train.npy` | Training features (scaled) |
| `X_val.npy` | Validation features (scaled) |
| `X_test.npy` | Test features (scaled) |
| `y_train.npy` | Training labels |
| `y_val.npy` | Validation labels |
| `y_test.npy` | Test labels |

> ⚠️ The first cell will ask you to authenticate with Google Drive — this is needed to save the preprocessed files. If you prefer not to connect Drive, the notebook still runs fine and you can download the arrays manually.
