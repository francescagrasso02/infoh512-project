"""
utils.py — Shared helpers for the fairness analysis (Levels 1-3).

Single source of truth for all DiCE-related helpers and project utilities:
    - get_project_paths()     : auto-detects local vs Colab and returns paths
    - load_artifacts()        : centralised pickle loader
    - load_raw_with_split()   : recovers the human-readable demographics
    - build_recourse_dataset(): runs DiCE on ALL predicted-attrition employees

Authors: Kamila (DiCE helpers), Matteo (fairness / FACTS analysis)
"""

import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# _________________________________________________________________________________
# PARALLEL WORKER STATE
# These must be module-level so worker processes (spawned on Windows) can
# import them from utils.py without touching __main__ (the notebook).

_worker_state = {}

def _init_worker(wrapper, dice_df, feature_names, method):
    """
    This function is called once per worker process; builds the DiCE explainer in-process.
    The Args are: 
        - wrapper: the model wrapper
        - dice_df: the dataframe for DiCE
        - feature_names: list of feature names
        - method: the DiCE method to use
    """

    import dice_ml
    global _worker_state
    dice_data = dice_ml.Data(
        dataframe=dice_df,
        continuous_features=list(feature_names),
        outcome_name="Attrition",
    )
    dice_model = dice_ml.Model(model=wrapper, backend="sklearn", model_type="classifier")
    _worker_state["explainer"] = dice_ml.Dice(dice_data, dice_model, method=method)
    _worker_state["method"] = method


def _process_one_employee(task):
    """
    This function is called by each worker process for each employee row it needs to process.
    It generates counterfactuals for one at-risk employee, computes the recourse outcome and cost,
    and returns a dictionary with the results.
    The Args are:
        - task: a tuple containing all the necessary information to process one employee, including:
            idx: the index of the employee in the dataframe
            q_enc_list: the encoded feature values of the employee as a list
            proba_val: the predicted probability of attrition for the employee
            salary_now: the current salary of the employee
            feats_to_vary: list of features that are allowed to vary in the counterfactuals
            permitted: a dict specifying the permitted range for each feature
            n_cf: the number of counterfactuals to generate
            seed: the random seed for reproducibility
            feature_names: list of all feature names
            feat_range_list: list of feature ranges for normalizing L1 cost
            raw_row_dict: a dict containing the raw, human-readable values for demographics and true attrition
    """

    (idx, q_enc_list, proba_val, salary_now, feats_to_vary, permitted,
     n_cf, seed, feature_names, feat_range_list, raw_row_dict) = task

    exp = _worker_state["explainer"]

    feat_range = pd.Series(feat_range_list, index=feature_names)
    q_enc = pd.DataFrame([q_enc_list], columns=feature_names)
    q_row = q_enc.iloc[0]

    cf_df = None
    method = _worker_state.get("method", "random")
    seed_kwarg = {"random_seed": seed} if method == "random" else {}
    try:
        cf = exp.generate_counterfactuals(
            q_enc,
            total_CFs=n_cf,
            desired_class=0,
            features_to_vary=feats_to_vary,
            permitted_range=permitted,
            verbose=False,
            **seed_kwarg,
        )
        cf_df = cf.cf_examples_list[0].final_cfs_df
    except Exception:
        pass

    n_found = 0 if cf_df is None else len(cf_df)
    recourse_found = n_found > 0
    l1 = mean_l1(q_row, cf_df, feature_names, feat_range) if recourse_found else float("nan")

    feat_changes = {}
    if recourse_found:
        assert cf_df is not None
        for _, cf_row in cf_df.iterrows():
            for f in changed_features(q_row, cf_row, feature_names):
                feat_changes[f] = feat_changes.get(f, 0) + 1

    return {
        "emp_id":         int(idx),
        "Gender":         raw_row_dict["Gender"],
        "AgeGroup":       raw_row_dict["AgeGroup"],
        "Age":            int(raw_row_dict["Age"]),
        "MaritalStatus":  raw_row_dict["MaritalStatus"],
        "Department":     raw_row_dict["Department"],
        "true_attrition": int(raw_row_dict["true_attrition"]),
        "predicted_risk": round(float(proba_val) * 100, 1),
        "n_cf_found":     n_found,
        "recourse_found": recourse_found,
        "L1_cost":        round(l1, 4) if recourse_found else np.nan,
        "changed_feats":  ";".join(sorted(feat_changes.keys())),
    }

# __________________________________________________________________________
# CONSTANTS  single source of truth
# Here are all the fixed values for the project in one place, so we can import them in the notebooks without duplication.

SEED_PREPROCESSING = 42      # the split seed used in 1_Preprocessing.ipynb
SEED_DICE          = 48      # the seed used by the models / DiCE notebooks

CATEGORICAL_FEATURES = [
    "BusinessTravel", 
    "Department", 
    "EducationField",
    "Gender",
    "JobRole", 
    "MaritalStatus", 
    "OverTime",
    "Education", 
    "EnvironmentSatisfaction", 
    "JobInvolvement",
    "JobLevel", 
    "JobSatisfaction", 
    "PerformanceRating",
    "RelationshipSatisfaction", 
    "StockOptionLevel", 
    "WorkLifeBalance",
]

IMMUTABLE_DEMOGRAPHIC = [
    "Age", 
    "Gender", 
    "MaritalStatus", 
    "EducationField", 
    "Department"]

IMMUTABLE_TIME = [
    "YearsAtCompany", 
    "YearsInCurrentRole", 
    "YearsSinceLastPromotion",
    "YearsWithCurrManager", 
    "TotalWorkingYears", 
    "NumCompaniesWorked",
]
IMMUTABLE_HISTORICAL = [
    "PercentSalaryHike", 
    "PerformanceRating", 
    "Education",
    "HourlyRate", 
    "DailyRate", 
    "MonthlyRate",
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
THRESHOLD_RF  = 0.489 # Remember to change this after you run the models
THRESHOLD_XGB = 0.592 # Remember to change this after you run the models


CSV_URL = (
    "https://raw.githubusercontent.com/francescagrasso02/"
    "infoh512-project/main/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# ____________________________________________________________________________
## PATH DETECTION  
# It automatically detects the environment where the code is running, since we are using Visual Studio
# and Google Colab, it is easier to set this up.

def get_project_paths():
    """
    This function auto-detects whether we are running on Google Colab or locally (VSCode),
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


# ___________________________________________________________________________________
# MODEL WRAPPER 
# For DiCE, we need to provide the data to the model. However, the model requires the data to be already scaled with StandardScaler.
# This wrapper is activated between calls to 'predict' and 'predict_proba',
# it applies the scaler automatically.
class ScaledModelWrapper:
    """Wraps a sklearn/XGBoost model + its scaler so DiCE can pass unscaled data."""

    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = list(feature_names)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        Xdf = pd.DataFrame(X, columns=self.feature_names)  # add names 
        return self.model.predict(self.scaler.transform(Xdf))

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        Xdf = pd.DataFrame(X, columns=self.feature_names)  # add names
        return self.model.predict_proba(self.scaler.transform(Xdf))


# ___________________________________________________________________________________
# DiCE Helpers
# This section contains helper functions for the DiCE algorithm. 

def decode_row(row, le_dict):
    """
    This function takes a row of the encoded dataframe and a dictionary of label encoders, and it
    converts integer-encoded categoricals back to readable labels.

    The Args are:
        - row: a pandas Series representing a single row of the encoded dataframe
        - le_dict: a dictionary where keys are column names and values are LabelEncoder objects
    """
    row = row.copy()
    for col, le in le_dict.items():
        if col in row.index:
            try:
                row[col] = le.inverse_transform([int(float(row[col]))])[0]
            except Exception:
                pass
    return row
    # The return is a pandas Series with the same index as the input row, but with categorical columns decoded to their original labels.


def mean_l1(q_row, cf_df, feature_names, feat_range):
    """
    This function computes the mean normalised L1 distance between a query row and its counterfactuals.
    
    The Args are:
        - q_row: a pandas Series representing the original query row (encoded)
        - cf_df: a pandas DataFrame containing the counterfactual rows (encoded)
        - feature_names: a list of feature names to consider in the distance calculation
        - feat_range: a pandas Series containing the range (max - min) of each feature in the training data, used for normalisation
    """
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
    # The return is the mean normalised L1 distance
    # where each feature's contribution is 0 if it didn't change
    # or the absolute difference divided by the feature's range if it did change.

def changed_features(q_row, cf_row, feature_names, tol=0.5):
    """
    This function identifies which features differ between a query row and a counterfactual row, 
    considering a tolerance for numerical features.
    
    The Args are:
        - q_row: a pandas Series representing the original query row (encoded)
        - cf_row: a pandas Series representing a counterfactual row (encoded)
        - feature_names: a list of feature names to check for changes
        - tol: a tolerance value for numerical features to consider them unchanged if the absolute difference is within this threshold
    """
    changed = []
    for f in feature_names:
        try:
            if abs(float(q_row[f]) - float(cf_row[f])) > tol:
                changed.append(f)
        except Exception:
            if str(q_row[f]) != str(cf_row[f]):
                changed.append(f)
    return changed
    # The return is a list of feature names that are considered changed between the query and counterfactual rows.


# _________________________________________________________________________________
# ARTIFACT LOADING
# These are the functions in charge of loading the models and their dependencies from a path. 
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_artifacts(models_dir):
    """
    This function loads the model and its associated artifacts from a specified directory.

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

# __________________________________________________________________________________
# DATA LOADING — replays the preprocessing to recover readable demographics

def load_raw_with_split(csv_url=CSV_URL, seed=SEED_PREPROCESSING):
    """
    This function reloads the original CSV and rebuilds the label-encoded frame that the
    preprocessing notebook produced; WITHOUT scaling, so the models can apply
    their own saved scalers at prediction time.

    Note: we use a fresh LabelEncoder per column here (same as notebook 1),
    which matches the encoding order in the saved feature_names.pkl.

    Returns:
        df_raw       : original CSV with human-readable strings + AgeGroup column
        df_encoded   : label-encoded, unscaled numeric frame (same column order as feature_names.pkl)
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
    # The return values are:
    # - df_raw: a pandas DataFrame containing the original CSV data with human-readable strings and an added 'AgeGroup' column
    # - df_encoded: a pandas DataFrame containing the label-encoded, unscaled numeric data (same column order as feature_names.pkl)

# _________________________________________________________________________________
# RECOURSE DATASET BUILDER — the bridge between DiCE and FACTS

def build_recourse_dataset(
    explainer, wrapper, df_encoded, df_raw, feature_names,
    le_dict, n_cf=3, seed=SEED_DICE, threshold=0.20,
    checkpoint_path=None, verbose=True,
    dice_df=None, n_jobs=1, dice_method="random",
    max_salary_increase=None,
):
    """
    This function builds a recourse dataset by running DiCE on every employee
    the model predicts as at-risk (proba >= threshold), recording the recourse
    outcome, cost, and which features changed, and attaching the readable
    demographics from df_raw.
    
    Pass dice_df (the DataFrame used to build dice_ml.Data) and n_jobs > 1 (or
    n_jobs=-1 for all CPU cores) to parallelise across employees. Each worker
    process builds its own DiCE explainer once, then processes many employees.
    If the Parallel path is not used, it runs sequentially in the same process
    (useful for debugging or if you have a small number of employees to process).

    The Args are: 
        - explainer: the DiCE explainer object (built on the same model wrapper)
        - wrapper: the model wrapper (needed for parallel workers)
        - df_encoded: the label-encoded, unscaled DataFrame used for predictions
        - df_raw: the original DataFrame with human-readable values (for demographics)
        - feature_names: the list of feature names (same order as model input)
        - le_dict: the dictionary of LabelEncoders used to decode demographics for the output
        - n_cf: the number of counterfactuals to generate per employee
        - seed: the random seed for reproducibility
        - threshold: the probability threshold to consider an employee at-risk
        - checkpoint_path: if provided, intermediate results are saved to this CSV after every 25 employees
        - verbose: whether to print progress messages
        - dice_df: the DataFrame used to build the dice_ml.Data object (required for parallel workers)
        - n_jobs: the number of parallel worker processes to use (default 1, -1 for all cores)
        - dice_method: the DiCE method to use ("random", "kdtree", etc.) — must be the same as the one used to build the explainer
    """
    feat_range = (
        df_encoded[feature_names].max() - df_encoded[feature_names].min()
    ).replace(0, 1)

    X_all       = df_encoded[feature_names].astype("float64")
    proba       = wrapper.predict_proba(X_all)[:, 1]
    at_risk_idx = np.where(proba >= threshold)[0]

    if verbose:
        print(f"Employees flagged at-risk (proba >= {threshold}): {len(at_risk_idx)}")

    salary_cap = max_salary_increase if max_salary_increase is not None else MAX_SALARY_INCREASE

    # parallel path ________________________________________________________
    if n_jobs != 1 and dice_df is not None:
        max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
        feat_range_list = feat_range.tolist()
        feats_to_vary   = [f for f in feature_names if f not in IMMUTABLE]

        tasks = []
        for idx in at_risk_idx:
            q_row       = X_all.iloc[idx]
            salary_now  = float(q_row["MonthlyIncome"])
            salary_ceil = round(salary_now * (1 + salary_cap))
            permitted   = {**PERMITTED_RANGE, "MonthlyIncome": [salary_now, salary_ceil]}
            raw         = df_raw.iloc[idx]
            raw_dict    = {
                "Gender":         raw["Gender"],
                "AgeGroup":       raw["AgeGroup"],
                "Age":            int(raw["Age"]),
                "MaritalStatus":  raw["MaritalStatus"],
                "Department":     raw["Department"],
                "true_attrition": int(raw["Attrition"]),
            }
            tasks.append((
                int(idx), q_row.tolist(), float(proba[idx]),
                salary_now, feats_to_vary, permitted,
                n_cf, seed, list(feature_names), feat_range_list, raw_dict,
            ))

        records_dict: dict[int, dict] = {}
        completed = 0

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(wrapper, dice_df, list(feature_names), dice_method),
        ) as executor:
            futures = {executor.submit(_process_one_employee, t): i
                       for i, t in enumerate(tasks)}
            for future in as_completed(futures):
                i = futures[future]
                records_dict[i] = future.result()
                completed += 1
                if verbose and completed % 25 == 0:
                    print(f"  processed {completed} / {len(tasks)}")
                if checkpoint_path and completed % 25 == 0:
                    pd.DataFrame(list(records_dict.values())).to_csv(
                        checkpoint_path, index=False
                    )

        result = pd.DataFrame([records_dict[i] for i in range(len(tasks))])
        if checkpoint_path:
            result.to_csv(checkpoint_path, index=False)
            if verbose:
                print(f"Saved -> {checkpoint_path}")
        return result

    # sequential path ________________________________________________________
    records = []
    processed_ids = set()

    if checkpoint_path and os.path.exists(checkpoint_path):
        try: 
            existing = pd.read_csv(checkpoint_path)
            if not existing.empty and "emp_id" in existing.columns:
                records = existing.to_dict("records")
                processed_ids = set(existing["emp_id"].tolist())
                if verbose:
                    print(f"  ✓ Resuming: {len(processed_ids)} employees already processed, "
                      f"{len(at_risk_idx) - len(processed_ids)} remaining")
        except Exception as e:
            if verbose:
                print(f"  [warn] Failed to read checkpoint: {e}")
   
    for count, idx in enumerate(at_risk_idx):
        if int(idx) in processed_ids:
            continue
        q_enc = X_all.iloc[[idx]].copy()
        q_row = q_enc.iloc[0]

        salary_now    = float(q_row["MonthlyIncome"])
        salary_ceil   = round(salary_now * (1 + salary_cap))
        permitted     = {**PERMITTED_RANGE, "MonthlyIncome": [salary_now, salary_ceil]}
        feats_to_vary = [f for f in feature_names if f not in IMMUTABLE]

        cf_df = None
        seed_kwarg = {"random_seed": seed} if dice_method == "random" else {}

        try:
            cf    = explainer.generate_counterfactuals(
                q_enc,
                total_CFs        = n_cf,
                desired_class    = 0,
                features_to_vary = feats_to_vary,
                permitted_range  = permitted,
                **seed_kwarg,
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
    # The return is a pandas DataFrame where each row corresponds to an at-risk employee and contains:
    # - emp_id: the index of the employee in the original dataframe (df_raw)
    # - Gender, AgeGroup, Age, MaritalStatus, Department: the demographics of the employee
    # - true_attrition: the actual attrition label (0 or 1) from df_raw
    # - predicted_risk: the predicted probability of attrition (as a percentage) from the model
    # - n_cf_found: the number of counterfactuals found by DiCE for this employee
    # - recourse_found: a boolean indicating whether at least one counterfactual was found for this employee
    # - L1_cost: the mean normalised L1 distance of the counterfactuals from the original employee (NaN if no counterfactuals found)
    # - changed_feats: a semicolon-separated string listing which features were changed in the counterfactuals (empty if no counterfactuals found) 
    