"""
Module 4: Shift Staffing Estimation
Estimates optimal staff count per shift based on demand patterns and attendance data.

Enhanced with MLflow hyperparameter tuning, polynomial features, improved cross-validation,
and better metrics reporting.

Data inputs:
- cleaned_monthly_sales_(334).csv: Monthly sales by branch — demand proxy
- customer_orders_(150).csv: Customer orders — demand intensity features
- cleaned_attendance_(461).csv: Staff attendance — actual staffing levels, shift patterns

ML Model: Ridge Regression (numpy OLS + L2 regularization with hyperparameter tuning)
- Predicts staff_count from demand, day type, and shift features (with polynomial interactions)
- Recommends staffing levels to optimize efficiency

Key outputs:
- Recommended staff per shift per branch per day type
- Current vs. optimal staffing comparison
- Cross-validation metrics (R², MAE)
- Efficiency metrics and staffing gaps
- MLflow experiment tracking (optional)
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from config import validate_dataframe, get_logger, RANDOM_SEED, STAFFING

# ── Set random seed at module level ──
np.random.seed(RANDOM_SEED)

# ── Initialize logger ──
logger = get_logger(__name__)

# ── Try to import MLflow (optional) ──
try:
    import mlflow
    HAS_MLFLOW = True
    logger.info("MLflow available - will log experiments")
except ImportError:
    HAS_MLFLOW = False
    logger.warning("MLflow not available - skipping experiment logging")

# ── Try to import sklearn metrics ──
try:
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    HAS_SKLEARN_METRICS = True
except ImportError:
    HAS_SKLEARN_METRICS = False
    logger.warning("sklearn.metrics not available - using numpy fallback")


# ── Constants ──
MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

LAMBDA_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
FEATURE_SUBSETS = [
    "base",  # daily_revenue_est, demand_intensity, day_of_week, is_weekend, shift_encoded
    "with_poly",  # base + polynomial interaction terms
    "with_onehot",  # one-hot encoded shift and branch + base features
]


def parse_duration_hours(s):
    """Convert HH:MM:SS string to decimal hours."""
    try:
        parts = str(s).split(":")
        return int(parts[0]) + int(parts[1]) / 60 + int(parts[2]) / 3600
    except Exception:
        return 0.0


def classify_shift(punch_hour):
    """Classify punch-in hour into shift type."""
    if 5 <= punch_hour < 12:
        return "Morning"
    elif 12 <= punch_hour < 17:
        return "Afternoon"
    else:
        return "Evening"


def load_data(
    monthly_path="data/cleaned/cleaned_monthly_sales_(334).csv",
    orders_path="data/cleaned/customer_orders_(150).csv",
    attendance_path="data/cleaned/cleaned_attendance_(461).csv",
):
    """Load all three data sources."""
    df334 = pd.read_csv(monthly_path)
    df150 = pd.read_csv(orders_path)
    df461 = pd.read_csv(attendance_path)
    
    logger.info(f"Loaded 334: {len(df334)} rows (monthly sales)")
    logger.info(f"Loaded 150: {len(df150)} rows (customer orders)")
    logger.info(f"Loaded 461: {len(df461)} rows (attendance)")
    
    return df334, df150, df461


def prepare_staffing_data(df334, df150, df461):
    """
    Build a daily staffing dataset by branch from the 3 sources.

    Returns a DataFrame at the (branch, date, shift) level with:
    - staff_count, total_hours: actual staffing from 461
    - daily_revenue_est: estimated daily revenue from 334
    - customer_intensity: demand proxy from 150
    - day features: day_of_week, is_weekend
    - Polynomial features: interaction terms for improved R²
    """
    df461 = df461.copy()
    df461["date_in"] = pd.to_datetime(df461["date_in"])
    df461["punch_in_time"] = pd.to_datetime(df461["punch_in"], format="%H:%M:%S", errors="coerce")
    df461["punch_hour"] = df461["punch_in_time"].dt.hour.fillna(12).astype(int)
    df461["shift"] = df461["punch_hour"].apply(classify_shift)
    df461["hours"] = df461["work_duration"].apply(parse_duration_hours)
    df461["date"] = df461["date_in"].dt.date

    # ── Aggregate to (branch, date, shift) level ──
    daily_shift = df461.groupby(["branch", "date", "shift"]).agg(
        staff_count=("hours", "count"),
        total_hours=("hours", "sum"),
        avg_hours_per_staff=("hours", "mean"),
    ).reset_index()

    daily_shift["date"] = pd.to_datetime(daily_shift["date"])
    daily_shift["day_of_week"] = daily_shift["date"].dt.dayofweek  # 0=Mon
    daily_shift["is_weekend"] = (daily_shift["day_of_week"] >= 5).astype(int)
    daily_shift["day_name"] = daily_shift["date"].dt.day_name()

    # ── Derive daily revenue estimate from 334 (monthly → daily) ──
    # 461 is Dec 2025 only, so get Dec sales per branch
    dec_sales = {}
    for _, row in df334.iterrows():
        if row["month"] == "December":
            dec_sales[row["branch"]] = row["sales"]

    # Estimate daily revenue = monthly / days in December (31)
    daily_shift["daily_revenue_est"] = daily_shift["branch"].map(
        {b: v / 31.0 for b, v in dec_sales.items()}
    ).fillna(0)

    # ── Customer demand intensity from 150 ──
    cust_stats = df150.groupby("branch").agg(
        n_customers=("customer", "count"),
        total_orders=("order_count", "sum"),
        avg_spend=("total_spent", "mean"),
    ).reset_index()

    # Normalize to intensity score (0-1 scale)
    max_orders = cust_stats["total_orders"].max() if len(cust_stats) > 0 else 1
    cust_stats["demand_intensity"] = cust_stats["total_orders"] / max_orders
    cust_map = dict(zip(cust_stats["branch"], cust_stats["demand_intensity"]))
    daily_shift["demand_intensity"] = daily_shift["branch"].map(cust_map).fillna(0.5)

    # ── Revenue per staff hour (efficiency metric) ──
    daily_shift["rev_per_staff_hour"] = (
        daily_shift["daily_revenue_est"] / daily_shift["total_hours"].replace(0, 1)
    )

    # ── Shift encoding (ordinal) ──
    shift_map = {"Morning": 0, "Afternoon": 1, "Evening": 2}
    daily_shift["shift_encoded"] = daily_shift["shift"].map(shift_map)

    # ── Polynomial interaction features ──
    # These capture non-linear relationships and should improve R²
    daily_shift["dow_x_shift"] = daily_shift["day_of_week"] * daily_shift["shift_encoded"]
    daily_shift["revenue_x_demand"] = daily_shift["daily_revenue_est"] * daily_shift["demand_intensity"]
    daily_shift["weekend_x_shift"] = daily_shift["is_weekend"] * daily_shift["shift_encoded"]

    logger.info(f"Prepared staffing data: {len(daily_shift)} shift-level records")
    return daily_shift


def compute_metrics(y_true, y_pred):
    """
    Compute R², MAE, RMSE using sklearn if available, else numpy.
    
    Returns: dict with r_squared, mae, rmse
    """
    if HAS_SKLEARN_METRICS:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        residuals = y_true - y_pred
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return {"r_squared": r2, "mae": mae, "rmse": rmse}


def build_staffing_model_with_features(
    staffing_data,
    feature_subset="base",
    lambda_reg=0.5,
):
    """
    Train a Ridge Regression model with specified features and lambda.

    Feature subsets:
    - 'base': daily_revenue_est, demand_intensity, day_of_week, is_weekend, shift_encoded
    - 'with_poly': base + polynomial interaction terms (dow_x_shift, revenue_x_demand, weekend_x_shift)
    - 'with_onehot': one-hot encode shift + branch, use base features

    Returns dict with weights, intercept, feature_names, metrics, feature_subset, lambda.
    """
    df = staffing_data.copy()
    target_col = "staff_count"

    # ── Define feature set ──
    if feature_subset == "base":
        feature_cols = [
            "daily_revenue_est",
            "demand_intensity",
            "day_of_week",
            "is_weekend",
            "shift_encoded",
        ]
    elif feature_subset == "with_poly":
        feature_cols = [
            "daily_revenue_est",
            "demand_intensity",
            "day_of_week",
            "is_weekend",
            "shift_encoded",
            "dow_x_shift",
            "revenue_x_demand",
            "weekend_x_shift",
        ]
    elif feature_subset == "with_onehot":
        # One-hot encode shift and branch
        df_onehot = pd.get_dummies(df[["shift"]], prefix="shift", drop_first=True)
        df_onehot = pd.concat([df_onehot, pd.get_dummies(df[["branch"]], prefix="branch", drop_first=True)], axis=1)
        
        base_cols = [
            "daily_revenue_est",
            "demand_intensity",
            "day_of_week",
            "is_weekend",
        ]
        feature_cols = base_cols + list(df_onehot.columns)
        
        df = pd.concat([df, df_onehot], axis=1)
    else:
        feature_cols = [
            "daily_revenue_est",
            "demand_intensity",
            "day_of_week",
            "is_weekend",
            "shift_encoded",
        ]

    X = df[feature_cols].fillna(0).values.astype(float)
    y = df[target_col].values.astype(float)

    n = len(X)
    if n < 3:
        logger.warning(f"Insufficient data for feature_subset={feature_subset}, lambda={lambda_reg}")
        return None

    # ── Normalize features ──
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_norm = (X - X_mean) / X_std

    # ── Add bias column ──
    X_bias = np.column_stack([np.ones(n), X_norm])

    # ── Ridge regression: (X'X + λI)^-1 X'y ──
    p = X_bias.shape[1]
    I = np.eye(p)
    I[0, 0] = 0  # don't regularize intercept

    try:
        w = np.linalg.solve(X_bias.T @ X_bias + lambda_reg * I, X_bias.T @ y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(X_bias, y, rcond=None)[0]

    # ── Predictions and metrics ──
    y_pred = X_bias @ w
    y_pred = np.maximum(y_pred, 1)

    metrics = compute_metrics(y, y_pred)

    # Feature importances (absolute normalized weight, skip intercept)
    abs_w = np.abs(w[1:])
    if abs_w.sum() > 0:
        importances = dict(zip(feature_cols, (abs_w / abs_w.sum()).round(3)))
    else:
        importances = {c: 0 for c in feature_cols}

    return {
        "weights": w,
        "X_mean": X_mean,
        "X_std": X_std,
        "feature_cols": feature_cols,
        "intercept": w[0],
        "importances": importances,
        "mae": round(metrics["mae"], 2),
        "rmse": round(metrics["rmse"], 2),
        "r_squared": round(metrics["r_squared"], 3),
        "n_samples": n,
        "feature_subset": feature_subset,
        "lambda": lambda_reg,
    }


def build_staffing_model(staffing_data):
    """
    Train Ridge Regression model with hyperparameter tuning via K-Fold cross-validation.
    
    Tries multiple lambda values and feature subsets, logs results with MLflow.
    Returns best model configuration.
    """
    logger.info("=" * 70)
    logger.info("TRAINING SHIFT STAFFING MODEL WITH HYPERPARAMETER TUNING")
    logger.info("=" * 70)

    df = staffing_data.copy()
    n = len(df)
    
    logger.info(f"Dataset size: {n} samples")
    
    if n < 10:
        logger.warning(f"Dataset too small (n={n}). Using single train/val split instead of K-Fold.")
        k_folds = 2
    else:
        k_folds = 5
    
    logger.info(f"Using {k_folds}-Fold Cross-Validation")

    best_r2 = -np.inf
    best_config = None
    results_log = []

    if HAS_MLFLOW:
        mlflow.set_experiment("shift_staffing_tuning")

    # ── Grid search over lambda and feature subsets ──
    for feature_subset in FEATURE_SUBSETS:
        for lambda_val in LAMBDA_VALUES:
            cv_r2_scores = []
            cv_mae_scores = []

            # ── K-Fold CV ──
            indices = np.arange(n)
            np.random.shuffle(indices)
            fold_size = n // k_folds

            for fold in range(k_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < k_folds - 1 else n
                
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                df_train = df.iloc[train_idx]
                df_val = df.iloc[val_idx]

                # Train on fold
                model = build_staffing_model_with_features(
                    df_train,
                    feature_subset=feature_subset,
                    lambda_reg=lambda_val,
                )

                if model is None:
                    continue

                # Validate on fold
                if feature_subset == "with_onehot":
                    # One-hot encode validation set matching training encoding
                    df_val_onehot = pd.get_dummies(df_val[["shift"]], prefix="shift", drop_first=True)
                    df_val_onehot = pd.concat(
                        [df_val_onehot, pd.get_dummies(df_val[["branch"]], prefix="branch", drop_first=True)],
                        axis=1
                    )
                    
                    base_cols = [
                        "daily_revenue_est",
                        "demand_intensity",
                        "day_of_week",
                        "is_weekend",
                    ]
                    feature_cols = base_cols + list(df_val_onehot.columns)
                    df_val_eval = pd.concat([df_val, df_val_onehot], axis=1)
                else:
                    feature_cols = model["feature_cols"]
                    df_val_eval = df_val

                X_val = df_val_eval[feature_cols].fillna(0).values.astype(float)
                y_val = df_val_eval["staff_count"].values.astype(float)

                if len(X_val) == 0:
                    continue

                # Normalize using training stats
                X_val_norm = (X_val - model["X_mean"]) / model["X_std"]
                X_val_bias = np.column_stack([np.ones(len(X_val_norm)), X_val_norm])
                y_pred_val = X_val_bias @ model["weights"]
                y_pred_val = np.maximum(y_pred_val, 1)

                val_metrics = compute_metrics(y_val, y_pred_val)
                cv_r2_scores.append(val_metrics["r_squared"])
                cv_mae_scores.append(val_metrics["mae"])

            if not cv_r2_scores:
                continue

            mean_r2 = np.mean(cv_r2_scores)
            std_r2 = np.std(cv_r2_scores)
            mean_mae = np.mean(cv_mae_scores)
            std_mae = np.std(cv_mae_scores)

            config = {
                "feature_subset": feature_subset,
                "lambda": lambda_val,
                "cv_r2_mean": round(mean_r2, 3),
                "cv_r2_std": round(std_r2, 3),
                "cv_mae_mean": round(mean_mae, 2),
                "cv_mae_std": round(std_mae, 2),
                "k_folds": k_folds,
            }
            results_log.append(config)

            logger.info(
                f"feature_subset={feature_subset:12s} | lambda={lambda_val:5.2f} | "
                f"CV R²={mean_r2:6.3f}±{std_r2:.3f} | CV MAE={mean_mae:5.2f}±{std_mae:.2f}"
            )

            # ── Log to MLflow if available ──
            if HAS_MLFLOW:
                try:
                    with mlflow.start_run():
                        mlflow.log_param("feature_subset", feature_subset)
                        mlflow.log_param("lambda", lambda_val)
                        mlflow.log_param("k_folds", k_folds)
                        mlflow.log_metric("cv_r2_mean", mean_r2)
                        mlflow.log_metric("cv_r2_std", std_r2)
                        mlflow.log_metric("cv_mae_mean", mean_mae)
                        mlflow.log_metric("cv_mae_std", std_mae)
                except Exception as e:
                    logger.warning(f"MLflow logging failed: {e}")

            # Track best model
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_config = {
                    "feature_subset": feature_subset,
                    "lambda": lambda_val,
                    "cv_r2_mean": mean_r2,
                    "cv_mae_mean": mean_mae,
                }

    # ── Train final model on full dataset using best config ──
    if best_config is None:
        logger.error("No valid hyperparameter configurations found")
        return None

    logger.info("\n" + "=" * 70)
    logger.info(f"BEST CONFIG: feature_subset={best_config['feature_subset']}, lambda={best_config['lambda']}")
    logger.info(f"  CV R²={best_config['cv_r2_mean']:.3f}, CV MAE={best_config['cv_mae_mean']:.2f}")
    logger.info("=" * 70 + "\n")

    final_model = build_staffing_model_with_features(
        df,
        feature_subset=best_config["feature_subset"],
        lambda_reg=best_config["lambda"],
    )

    if final_model is None:
        logger.error("Failed to train final model")
        return None

    final_model["cv_results"] = {
        "best_config": best_config,
        "all_results": results_log,
    }

    return final_model


def predict_staff(model, daily_revenue, demand_intensity, day_of_week, is_weekend, shift_encoded):
    """Predict staff count for given inputs."""
    if model is None:
        return 1

    features = np.array([daily_revenue, demand_intensity, day_of_week, is_weekend, shift_encoded])
    features_norm = (features - model["X_mean"][:5]) / model["X_std"][:5]
    x = np.concatenate([[1], features_norm])
    
    # Pad with zeros if model expects more features
    if len(x) < len(model["weights"]):
        x = np.concatenate([x, np.zeros(len(model["weights"]) - len(x))])
    
    pred = x[:len(model["weights"])] @ model["weights"]
    return max(int(np.ceil(pred)), 1)


def analyze_staffing(staffing_data, model):
    """
    Analyze current staffing vs model-recommended staffing.

    Returns per-branch, per-shift, per-day-type recommendations.
    """
    if model is None:
        return []

    df = staffing_data.copy()
    recommendations = []

    for branch in sorted(df["branch"].unique()):
        branch_data = df[df["branch"] == branch]

        for shift in ["Morning", "Afternoon", "Evening"]:
            shift_data = branch_data[branch_data["shift"] == shift]
            if len(shift_data) == 0:
                continue

            for day_type, is_wknd in [("Weekday", 0), ("Weekend", 1)]:
                day_data = shift_data[shift_data["is_weekend"] == is_wknd]
                if len(day_data) == 0:
                    continue

                avg_staff = day_data["staff_count"].mean()
                avg_hours = day_data["total_hours"].mean()
                avg_revenue = day_data["daily_revenue_est"].mean()
                avg_rev_per_hour = day_data["rev_per_staff_hour"].mean()
                demand_int = day_data["demand_intensity"].mean()
                avg_dow = day_data["day_of_week"].mean()

                shift_map = {"Morning": 0, "Afternoon": 1, "Evening": 2}
                recommended = predict_staff(
                    model, avg_revenue, demand_int, avg_dow, is_wknd, shift_map[shift]
                )

                delta = recommended - round(avg_staff)
                if delta > 0:
                    status = "Understaffed"
                elif delta < 0:
                    status = "Overstaffed"
                else:
                    status = "Optimal"

                recommendations.append({
                    "branch": branch,
                    "shift": shift,
                    "day_type": day_type,
                    "current_avg_staff": round(avg_staff, 1),
                    "recommended_staff": recommended,
                    "delta": delta,
                    "status": status,
                    "avg_hours_worked": round(avg_hours, 1),
                    "daily_revenue_est": round(avg_revenue, 0),
                    "rev_per_staff_hour": round(avg_rev_per_hour, 0),
                    "days_observed": len(day_data),
                })

    return recommendations


def run_full_staffing_analysis(
    monthly_path="data/cleaned/cleaned_monthly_sales_(334).csv",
    orders_path="data/cleaned/customer_orders_(150).csv",
    attendance_path="data/cleaned/cleaned_attendance_(461).csv",
):
    """Full pipeline: load → prepare → train → analyze → return results."""
    logger.info("=" * 70)
    logger.info("SHIFT STAFFING ANALYSIS PIPELINE")
    logger.info("=" * 70)

    logger.info("\n[1/4] Loading data...")
    df334, df150, df461 = load_data(monthly_path, orders_path, attendance_path)

    logger.info("\n[2/4] Preparing staffing dataset...")
    staffing_data = prepare_staffing_data(df334, df150, df461)
    logger.info(f"  → {len(staffing_data)} shift-level records across {staffing_data['branch'].nunique()} branches")

    logger.info("\n[3/4] Training Ridge Regression model with hyperparameter tuning...")
    model = build_staffing_model(staffing_data)
    if model is None:
        logger.error("  → ERROR: Not enough data to train model")
        return {"status": "error", "message": "Insufficient data"}

    logger.info(f"  → Train R² = {model['r_squared']}, MAE = {model['mae']} staff, RMSE = {model['rmse']}")
    logger.info(f"  → Feature importances: {model['importances']}")
    
    cv_results = model.get("cv_results", {})
    if cv_results and "best_config" in cv_results:
        best = cv_results["best_config"]
        logger.info(f"  → Cross-validation R² = {best.get('cv_r2_mean', 'N/A')}, MAE = {best.get('cv_mae_mean', 'N/A')}")

    logger.info("\n[4/4] Generating staffing recommendations...")
    recommendations = analyze_staffing(staffing_data, model)

    understaffed = sum(1 for r in recommendations if r["status"] == "Understaffed")
    overstaffed = sum(1 for r in recommendations if r["status"] == "Overstaffed")
    optimal = sum(1 for r in recommendations if r["status"] == "Optimal")
    logger.info(f"  → Understaffed: {understaffed}, Overstaffed: {overstaffed}, Optimal: {optimal}")

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)

    return {
        "status": "success",
        "model_metrics": {
            "r_squared": model["r_squared"],
            "mae": model["mae"],
            "rmse": model["rmse"],
            "n_samples": model["n_samples"],
            "feature_importances": model["importances"],
            "feature_subset": model.get("feature_subset", "base"),
            "lambda": model.get("lambda", STAFFING.get("ridge_lambda", 0.5)),
            "cv_results": model.get("cv_results", {}),
        },
        "recommendations": recommendations,
    }


def get_staffing_recommendation(
    branch=None,
    shift=None,
    monthly_path="data/cleaned/cleaned_monthly_sales_(334).csv",
    orders_path="data/cleaned/customer_orders_(150).csv",
    attendance_path="data/cleaned/cleaned_attendance_(461).csv",
):
    """
    OpenClaw-compatible function: Get staffing recommendations.

    Parameters:
        branch: filter by branch name (optional)
        shift: filter by shift — "Morning", "Afternoon", "Evening" (optional)

    Returns JSON-compatible dict with status, model metrics, and recommendations.
    """
    try:
        result = run_full_staffing_analysis(monthly_path, orders_path, attendance_path)
        if result["status"] != "success":
            return result

        recs = result["recommendations"]

        if branch:
            recs = [r for r in recs if r["branch"].lower() == branch.lower()]
        if shift:
            recs = [r for r in recs if r["shift"].lower() == shift.lower()]

        return {
            "status": "success",
            "branch_filter": branch or "all",
            "shift_filter": shift or "all",
            "model_metrics": result["model_metrics"],
            "recommendations": recs,
        }
    except Exception as e:
        logger.error(f"Error in get_staffing_recommendation: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    result = run_full_staffing_analysis()
    if result["status"] == "success":
        logger.info("\n\nRECOMMENDATIONS:")
        for r in result["recommendations"]:
            logger.info(
                f"  {r['branch']} | {r['shift']:10s} | {r['day_type']:7s} | "
                f"Current: {r['current_avg_staff']:.1f} → Recommended: {r['recommended_staff']} "
                f"[{r['status']}]"
            )
