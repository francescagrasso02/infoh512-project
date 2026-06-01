"""
utils.py — Shared helpers for the fairness analysis (Levels 1-3).

Most of the DiCE-related code below is REUSED verbatim from the team's
DiCE notebook (3_DiCE.ipynb, authored by Kamila). It is centralised here so
that the fairness notebooks import a single source of truth instead of
duplicating definitions. The only additions are:
    - get_project_paths()     : auto-detects local vs Colab and returns paths
    - load_artifacts()        : centralised pickle loader
    - load_raw_with_split()   : recovers the human-readable demographics
    - build_recourse_dataset(): runs DiCE on ALL predicted-attrition employees

Author of the new code: Matteo (fairness / FACTS analysis)
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (copied from the team's DiCE notebook — single source of truth)
# ──────────────────────────────────────────────────────────────────────────────
SEED_PREPROCESSING = 42      # the split seed used in 1_Preprocessing.ipynb
SEED_DICE          = 48      # the seed used by the models / DiCE notebooks

CATEGORICAL_FEATURES = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime",
    "Education", "EnvironmentSatisfaction", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance",
]

IMMUTABLE_DEMOGRAPHIC = ["Age", "Gender", "MaritalStatus", "EducationField", "Department"]
IMMUTABLE_TIME = [
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager", "TotalWorkingYears", "NumCompaniesWorked",
]
IMMUTABLE_HISTORICAL = [
    "PercentSalaryHike", "PerformanceRating", "Education",
    "HourlyRate", "DailyRate", "MonthlyRate",
]
IMMUTABLE = IMMUTABLE_DEMOGRAPHIC + IMMUTABLE_TIME + IMMUTABLE_HISTORICAL

PERMITTED_RANGE = {
    "JobLevel":                [1, 3],
    "StockOptionLevel":        [0, 2],
    "WorkLifeBalance":         [1, 4],
    "EnvironmentSatisfaction": [1, 4],
    "JobSatisfaction":         [1, 4],
    "JobInvolvement":          [1, 4],
    "RelationshipSatisfaction":[1, 4],
    "TrainingTimesLastYear":   [0, 6],
}
MAX_SALARY_INCREASE = 0.50

# Protected attributes we audit, and how we bucket the continuous ones.
PROTECTED_ATTRIBUTES = ["Gender", "AgeGroup", "MaritalStatus"]
AGE_THRESHOLD = 35   # AgeGroup = "Young" if Age < 35 else "Senior"

CSV_URL = (
    "https://raw.githubusercontent.com/francescagrasso02/"
    "infoh512-project/main/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)


# ──────────────────────────────────────────────────────────────────────────────
# PATH DETECTION  (NEW)
# ──────────────────────────────────────────────────────────────────────────────
def get_project_paths():
    """
    Auto-detects whether we are running on Google Colab or locally (VSCode),
    and returns a dict of all relevant project paths.

    Local layout assumed:
        infoh512-project-1/
            notebooks/        ← notebooks and utils.py live here
            data/
                models/RF/
                models/XGBoost/
                fairness/     ← created automatically

    On Colab, mounts Google Drive and looks for the shared
    Current_Trends_in_AI/ folder.
    """
    try:
        import google.colab  # noqa: F401
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)

        # Try the standard locations Francesca set up
        for candidate in [
            "/content/drive/MyDrive/Current_Trends_in_AI",
            "/content/drive/MyDrive/Shared with me/Current_Trends_in_AI",
            "/content/drive/Shareddrives/Current_Trends_in_AI",
        ]:
            if os.path.exists(candidate):
                base = candidate
                break
        else:
            base = "/content/drive/MyDrive/Current_Trends_in_AI"
            print(f"[warn] Could not find project folder on Drive. Using: {base}")

    except ImportError:
        # Running locally — notebooks/ is the CWD when a notebook is open in VSCode.
        # Go one level up to reach the project root.
        base = os.path.abspath(os.path.join(os.getcwd(), ".."))

    paths = {
        "BASE":     base,
        "DATA":     os.path.join(base, "data", "processed"),
        "RF":       os.path.join(base, "data", "models", "RF"),
        "XGBoost":  os.path.join(base, "data", "models", "XGBoost"),
        "FAIRNESS": os.path.join(base, "data", "fairness"),
    }

    # Create the fairness output dir if it doesn't exist yet.
    os.makedirs(paths["FAIRNESS"], exist_ok=True)

    return paths


# ──────────────────────────────────────────────────────────────────────────────
# MODEL WRAPPER  (copied verbatim from the team's DiCE notebook)
# ──────────────────────────────────────────────────────────────────────────────
class ScaledModelWrapper:
    """Wraps a sklearn/XGBoost model + its scaler so DiCE can pass unscaled data."""

    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = list(feature_names)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        return self.model.predict_proba(self.scaler.transform(X))


# ──────────────────────────────────────────────────────────────────────────────
# DiCE COST / DECODE HELPERS  (copied verbatim from the team's DiCE notebook)
# ──────────────────────────────────────────────────────────────────────────────
def decode_row(row, le_dict):
    """Convert integer-encoded categoricals back to readable labels."""
    row = row.copy()
    for col, le in le_dict.items():
        if col in row.index:
            try:
                row[col] = le.inverse_transform([int(float(row[col]))])[0]
            except Exception:
                pass
    return row


def mean_l1(q_row, cf_df, feature_names, feat_range):
    """Mean normalised L1 distance between a query row and its counterfactuals."""
    if cf_df is None or len(cf_df) == 0:
        return float("nan")
    dists = []
    for _, cf_row in cf_df.iterrows():
        d = sum(
            abs(float(q_row[f]) - float(cf_row[f])) / feat_range[f]
            if str(q_row[f]) != str(cf_row[f]) else 0
            for f in feature_names
        )
        dists.append(d)
    return sum(dists) / len(dists)


def changed_features(q_row, cf_row, feature_names, tol=0.5):
    """Return the list of feature names that differ between a query row and a CF."""
    changed = []
    for f in feature_names:
        try:
            if abs(float(q_row[f]) - float(cf_row[f])) > tol:
                changed.append(f)
        except Exception:
            if str(q_row[f]) != str(cf_row[f]):
                changed.append(f)
    return changed


# ──────────────────────────────────────────────────────────────────────────────
# ARTIFACT LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_artifacts(models_dir):
    """
    Load model + scaler + label_encoders + feature_names from a model directory.

    Expects the layout produced by the RF / XGBoost notebooks:
        models_dir/random_forest.pkl  (or xgboost.pkl)
        models_dir/scaler.pkl
        models_dir/label_encoders.pkl
        models_dir/feature_names.pkl

    Returns a dict with keys: model, scaler, label_encoders, feature_names.
    """
    model_file = None
    for candidate in ("random_forest.pkl", "xgboost.pkl"):
        p = os.path.join(models_dir, candidate)
        if os.path.exists(p):
            model_file = p
            break
    if model_file is None:
        raise FileNotFoundError(f"No model .pkl found in {models_dir}")

    return {
        "model":          load_pickle(model_file),
        "scaler":         load_pickle(os.path.join(models_dir, "scaler.pkl")),
        "label_encoders": load_pickle(os.path.join(models_dir, "label_encoders.pkl")),
        "feature_names":  load_pickle(os.path.join(models_dir, "feature_names.pkl")),
    }


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING — replays the preprocessing to recover readable demographics
# ──────────────────────────────────────────────────────────────────────────────
def load_raw_with_split(csv_url=CSV_URL, seed=SEED_PREPROCESSING):
    """
    Reloads the original CSV and rebuilds the label-encoded frame that the
    preprocessing notebook produced — WITHOUT scaling, so the models can apply
    their own saved scalers at prediction time.

    Note: we use a fresh LabelEncoder per column here (same as notebook 1),
    which matches the encoding order in the saved feature_names.pkl.

    Returns:
        df_raw       : original CSV with human-readable strings + AgeGroup column
        df_encoded   : label-encoded, unscaled numeric frame (same column order
                       as feature_names.pkl)
        feature_cols : ordered list of feature columns (excludes 'Attrition')
    """
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(csv_url)
    df = df.drop(columns=["EmployeeCount", "StandardHours", "EmployeeNumber", "Over18"])

    # Keep a human-readable copy before encoding.
    df_raw = df.copy()
    df_raw["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    df_raw["AgeGroup"]  = np.where(df_raw["Age"] < AGE_THRESHOLD, "Young", "Senior")

    # Encode the working copy exactly as notebook 1 did.
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    feature_cols = [c for c in df.columns if c != "Attrition"]

    return df_raw, df, feature_cols


# ──────────────────────────────────────────────────────────────────────────────
# RECOURSE DATASET BUILDER — the bridge between DiCE and FACTS
# ──────────────────────────────────────────────────────────────────────────────
def build_recourse_dataset(
    explainer, wrapper, df_encoded, df_raw, feature_names,
    le_dict, n_cf=3, seed=SEED_DICE, threshold=0.20,
    checkpoint_path=None, verbose=True,
):
    """
    Runs DiCE on EVERY employee the model predicts as at-risk (proba >= threshold),
    records the recourse outcome, cost, and which features changed, and attaches
    the readable demographics from df_raw.

    This is the key function that enables FACTS analysis on the full population
    rather than the 3-employee sample in Kamila's DiCE notebook.

    Output DataFrame columns:
        emp_id, Gender, AgeGroup, Age, MaritalStatus, Department,
        true_attrition, predicted_risk, n_cf_found, recourse_found,
        L1_cost, changed_feats

    Results are checkpointed to CSV every 25 employees so a crash doesn't
    lose all progress.
    """
    feat_range = (
        df_encoded[feature_names].max() - df_encoded[feature_names].min()
    ).replace(0, 1)

    X_all       = df_encoded[feature_names].astype("float64")
    proba       = wrapper.predict_proba(X_all)[:, 1]
    at_risk_idx = np.where(proba >= threshold)[0]

    if verbose:
        print(f"Employees flagged at-risk (proba >= {threshold}): {len(at_risk_idx)}")

    records = []
    for count, idx in enumerate(at_risk_idx):
        q_enc = X_all.iloc[[idx]].copy()
        q_row = q_enc.iloc[0]

        salary_now    = float(q_row["MonthlyIncome"])
        salary_ceil   = round(salary_now * (1 + MAX_SALARY_INCREASE))
        permitted     = {**PERMITTED_RANGE, "MonthlyIncome": [salary_now, salary_ceil]}
        feats_to_vary = [f for f in feature_names if f not in IMMUTABLE]

        cf_df = None
        try:
            cf    = explainer.generate_counterfactuals(
                q_enc,
                total_CFs        = n_cf,
                desired_class    = 0,
                features_to_vary = feats_to_vary,
                permitted_range  = permitted,
                random_seed      = seed,
                verbose          = False,
            )
            cf_df = cf.cf_examples_list[0].final_cfs_df
        except Exception as e:
            if verbose:
                print(f"  [warn] employee row {idx}: DiCE failed ({e})")

        n_found        = 0 if cf_df is None else len(cf_df)
        recourse_found = n_found > 0
        l1             = mean_l1(q_row, cf_df, feature_names, feat_range) if recourse_found else float("nan")

        feat_changes = {}
        if recourse_found:
            for _, cf_row in cf_df.iterrows():
                for f in changed_features(q_row, cf_row, feature_names):
                    feat_changes[f] = feat_changes.get(f, 0) + 1

        raw = df_raw.iloc[idx]
        records.append({
            "emp_id":         int(idx),
            "Gender":         raw["Gender"],
            "AgeGroup":       raw["AgeGroup"],
            "Age":            int(raw["Age"]),
            "MaritalStatus":  raw["MaritalStatus"],
            "Department":     raw["Department"],
            "true_attrition": int(raw["Attrition"]),
            "predicted_risk": round(float(proba[idx]) * 100, 1),
            "n_cf_found":     n_found,
            "recourse_found": recourse_found,
            "L1_cost":        round(l1, 4) if recourse_found else np.nan,
            "changed_feats":  ";".join(sorted(feat_changes.keys())),
        })

        if verbose and (count + 1) % 25 == 0:
            print(f"  processed {count + 1} / {len(at_risk_idx)}")
        if checkpoint_path and (count + 1) % 25 == 0:
            pd.DataFrame(records).to_csv(checkpoint_path, index=False)

    result = pd.DataFrame(records)
    if checkpoint_path:
        result.to_csv(checkpoint_path, index=False)
        if verbose:
            print(f"Saved -> {checkpoint_path}")
    return result