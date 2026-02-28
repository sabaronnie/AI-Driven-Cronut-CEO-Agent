"""
Module 2: Demand Forecasting by Branch
Predicts future sales/demand per branch using time series models.

Built from scratch using numpy (no sklearn/statsmodels/scipy dependency).

Data inputs:
- cleaned_monthly_sales_(334).csv: Monthly sales by branch (Aug-Dec 2025) — TARGET
- customer_orders_(150).csv: Customer orders — demand-side features (customer count, frequency, avg spend)
- cleaned_attendance_(461).csv: Staff attendance — supply-side features (staffing levels, work hours)

ML Models implemented (pure numpy):
1. Polynomial Regression with feature engineering + Ridge regularization
2. Exponential Smoothing (Holt's method with optimized hyperparameters)
3. Weighted Moving Average as baseline
4. Ensemble of all 3 models

Key outputs:
- Forecasted monthly sales per branch for next N months
- Model accuracy metrics (MAE, MAPE, RMSE, R²)
- Branch-level demand insights
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ── Config imports ──
from config import (
    validate_dataframe,
    get_logger,
    RANDOM_SEED,
    DEMAND,
    MONTH_ORDER,
    MONTH_NAMES,
)

# ── Set random seed for reproducibility ──
np.random.seed(RANDOM_SEED)

# ── Initialize logger ──
logger = get_logger("demand_forecasting")

# ── Try to import MLflow for experiment tracking ──
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.info("MLflow not available; experiment tracking disabled")

# ── Try to import sklearn metrics ──
try:
    from sklearn.metrics import mean_absolute_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_data(
    monthly_path="data/cleaned/cleaned_monthly_sales_(334).csv",
    orders_path="data/cleaned/customer_orders_(150).csv",
    attendance_path="data/cleaned/cleaned_attendance_(461).csv",
):
    """
    Load all three data sources for demand forecasting.
    Validates that all required columns are present.
    """
    df334 = pd.read_csv(monthly_path)
    df150 = pd.read_csv(orders_path)
    df461 = pd.read_csv(attendance_path)

    # ── Input validation ──
    validate_dataframe(df334, ["branch", "month", "year", "sales"], "df334 (monthly_sales)")
    validate_dataframe(df150, ["branch", "customer", "order_count", "total_spent"], "df150 (customer_orders)")
    validate_dataframe(df461, ["branch", "date_in", "punch_in", "work_duration"], "df461 (attendance)")

    logger.info(f"Loaded 334: {len(df334)} rows (monthly sales by branch)")
    logger.info(f"Loaded 150: {len(df150)} rows (customer orders)")
    logger.info(f"Loaded 461: {len(df461)} rows (staff attendance)")

    return df334, df150, df461


def prepare_time_series(df334):
    """
    Convert monthly sales data into a time-indexed series per branch.
    """
    df = df334.copy()
    df["month_num"] = df["month"].map(MONTH_ORDER)
    df = df.sort_values(["branch", "year", "month_num"])

    branch_series = {}
    for branch in df["branch"].unique():
        bdf = df[df["branch"] == branch].copy()
        bdf = bdf[bdf["sales"] > 0].reset_index(drop=True)
        bdf["t"] = np.arange(len(bdf))
        branch_series[branch] = bdf

    return branch_series


def extract_features(branch_df, df150, df461):
    """
    Engineer features for each branch from all 3 data sources.

    Features from 334: time index, month seasonality
    Features from 150: customer count, avg spend per customer, order frequency
    Features from 461: avg staff per day, avg work hours, staffing capacity
    """
    branch_name = branch_df["branch"].iloc[0]
    n = len(branch_df)

    # ── Time features from 334 ──
    t = branch_df["t"].values.astype(float)
    month_nums = branch_df["month_num"].values.astype(float)
    month_sin = np.sin(2 * np.pi * month_nums / 12)
    month_cos = np.cos(2 * np.pi * month_nums / 12)

    # ── Customer demand features from 150 ──
    branch_orders = df150[df150["branch"] == branch_name]
    n_customers = len(branch_orders)
    total_orders = branch_orders["order_count"].sum()
    total_spent = branch_orders["total_spent"].sum()
    avg_spend_per_customer = total_spent / n_customers if n_customers > 0 else 0
    avg_orders_per_customer = total_orders / n_customers if n_customers > 0 else 0
    repeat_rate = len(branch_orders[branch_orders["order_count"] > 1]) / n_customers if n_customers > 0 else 0

    # ── Staffing features from 461 ──
    branch_attendance = df461[df461["branch"] == branch_name]
    n_staff_records = len(branch_attendance)

    if n_staff_records > 0:
        # Parse work duration to hours
        def parse_duration(d):
            try:
                parts = str(d).split(":")
                return float(parts[0]) + float(parts[1]) / 60
            except Exception:
                return 0
        work_hours = branch_attendance["work_duration"].apply(parse_duration)
        avg_work_hours = work_hours.mean()
        total_work_hours = work_hours.sum()

        # Unique working days
        unique_days = branch_attendance["date_in"].nunique()
        avg_staff_per_day = n_staff_records / unique_days if unique_days > 0 else 0
    else:
        avg_work_hours = 0
        total_work_hours = 0
        avg_staff_per_day = 0
        unique_days = 0

    # ── Build feature matrix ──
    # With only 4-5 data points, keep features minimal to avoid overfitting
    small_n_threshold = DEMAND["small_n_threshold"]
    if n <= small_n_threshold:
        X = np.column_stack([
            t,
            month_sin,
        ])
        feature_names = ["t", "month_sin"]
    else:
        X = np.column_stack([
            t,
            t ** 2,
            month_sin,
            month_cos,
            np.full(n, n_customers),
            np.full(n, avg_staff_per_day),
        ])
        feature_names = [
            "t", "t_squared", "month_sin", "month_cos",
            "n_customers", "avg_staff_per_day",
        ]

    metadata = {
        "n_customers": n_customers,
        "total_orders": total_orders,
        "avg_spend_per_customer": avg_spend_per_customer,
        "avg_orders_per_customer": avg_orders_per_customer,
        "repeat_customer_rate": repeat_rate,
        "staff_records": n_staff_records,
        "avg_work_hours": avg_work_hours,
        "avg_staff_per_day": avg_staff_per_day,
        "total_staff_hours": total_work_hours,
        "working_days_recorded": unique_days,
    }

    return X, feature_names, metadata


# ═══════════════════════════════════════════════════
# ML Model 1: Polynomial Regression (pure numpy OLS)
# ═══════════════════════════════════════════════════

def fit_polynomial_regression(X, y, branch_name=None):
    """
    Fit multivariate linear regression using numpy least squares.
    Includes MLflow tracking of Ridge lambda tuning and Leave-One-Out CV.

    Solves: w = (X^T X + λI)^{-1} X^T y  (Ridge regression)
    """
    lambdas = DEMAND.get("ridge_lambda_tuning", [0.01, 0.05, 0.1, 0.5, 1.0])
    best_lambda = DEMAND["ridge_lambda"]
    best_mape = float("inf")
    best_weights = None
    best_cv_metrics = {}

    # ── MLflow run for Ridge lambda tuning ──
    if HAS_MLFLOW and branch_name:
        mlflow.start_run(run_name=f"poly_ridge_tuning_{branch_name}")
        mlflow.log_param("branch", branch_name)
        mlflow.log_param("n_samples", len(y))

    try:
        for lambda_reg in lambdas:
            X_aug = np.column_stack([np.ones(len(X)), X])

            # ── Leave-One-Out Cross-Validation (LOO-CV) ──
            loo_mapes = []
            loo_maes = []
            for i in range(len(X_aug)):
                X_train = np.vstack([X_aug[:i], X_aug[i+1:]])
                y_train = np.concatenate([y[:i], y[i+1:]])
                X_test = X_aug[i:i+1]
                y_test = y[i:i+1]

                XtX = X_train.T @ X_train + lambda_reg * np.eye(X_train.shape[1])
                Xty = X_train.T @ y_train

                try:
                    w = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    w = np.linalg.pinv(X_train) @ y_train

                y_pred_test = X_test @ w
                residual = y_test[0] - y_pred_test[0]
                loo_maes.append(abs(residual))
                loo_mapes.append(abs(residual / max(abs(y_test[0]), 1e-10)) * 100)

            cv_mape = np.mean(loo_mapes) if loo_mapes else float("inf")
            cv_mae = np.mean(loo_maes) if loo_maes else 0

            # ── Full training on all data for this lambda ──
            XtX = X_aug.T @ X_aug + lambda_reg * np.eye(X_aug.shape[1])
            Xty = X_aug.T @ y

            try:
                weights = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                weights = np.linalg.pinv(X_aug) @ y

            y_pred = X_aug @ weights
            residuals = y - y_pred

            mae = np.mean(np.abs(residuals))
            mape = np.mean(np.abs(residuals / np.maximum(np.abs(y), 1e-10))) * 100

            # ── Log to MLflow ──
            if HAS_MLFLOW and branch_name:
                with mlflow.start_run(nested=True, run_name=f"lambda_{lambda_reg}"):
                    mlflow.log_param("lambda", lambda_reg)
                    mlflow.log_metric("train_mape", mape)
                    mlflow.log_metric("train_mae", mae)
                    mlflow.log_metric("cv_mape", cv_mape)
                    mlflow.log_metric("cv_mae", cv_mae)

            if cv_mape < best_mape:
                best_mape = cv_mape
                best_lambda = lambda_reg
                best_weights = weights
                best_cv_metrics = {"cv_mape": cv_mape, "cv_mae": cv_mae}

        # Recalculate metrics with best lambda for final result
        X_aug = np.column_stack([np.ones(len(X)), X])
        y_pred = X_aug @ best_weights
        residuals = y - y_pred

        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        mape = np.mean(np.abs(residuals / np.maximum(np.abs(y), 1e-10))) * 100
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        if HAS_MLFLOW and branch_name:
            mlflow.log_param("best_lambda", best_lambda)
            mlflow.log_metric("final_mape", mape)
            mlflow.log_metric("final_mae", mae)
            mlflow.log_metric("final_rmse", rmse)
            mlflow.log_metric("final_r_squared", r_squared)
            mlflow.log_metric("cv_mape_best", best_cv_metrics.get("cv_mape", 0))

    finally:
        if HAS_MLFLOW and branch_name:
            mlflow.end_run()

    return {
        "weights": best_weights,
        "metrics": {"mae": mae, "rmse": rmse, "mape": mape, "r_squared": r_squared},
        "cv_metrics": best_cv_metrics,
        "predictions": y_pred,
        "residuals": residuals,
        "best_lambda": best_lambda,
    }


def predict_polynomial(X_future, weights):
    """Predict using trained polynomial regression weights."""
    X_aug = np.column_stack([np.ones(len(X_future)), X_future])
    return X_aug @ weights


# ═══════════════════════════════════════════════════
# ML Model 2: Holt's Exponential Smoothing (trend)
# ═══════════════════════════════════════════════════

def holts_exponential_smoothing(y, alpha=0.3, beta=0.1, periods=3):
    """
    Holt's double exponential smoothing for data with trend.
    Learns level and trend components from the time series.
    """
    n = len(y)
    if n < 2:
        return {"forecast": [y[-1]] * periods, "level": y[-1], "trend": 0,
                "metrics": {"mae": 0, "rmse": 0, "mape": 0}}

    level = y[0]
    trend = y[1] - y[0]
    fitted = [level + trend]

    for t in range(1, n):
        new_level = alpha * y[t] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
        trend = new_trend
        fitted.append(level + trend)

    forecast = [level + h * trend for h in range(1, periods + 1)]

    fitted = np.array(fitted[:n])
    residuals = y - fitted
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    mape = np.mean(np.abs(residuals / np.maximum(np.abs(y), 1e-10))) * 100

    return {
        "forecast": forecast,
        "fitted": fitted,
        "level": level,
        "trend": trend,
        "metrics": {"mae": mae, "rmse": rmse, "mape": mape},
        "alpha": alpha,
        "beta": beta,
    }


def optimize_holt_params(y, branch_name=None, periods=3):
    """
    Grid search over alpha and beta to find best Holt's parameters.
    Logs to MLflow for experiment tracking.
    This is hyperparameter optimization — a core ML technique.
    """
    best_mape = float("inf")
    best_result = None
    best_params = (0.3, 0.1)

    holt_alpha_range = DEMAND.get("holt_alpha_range", (0.1, 0.9, 0.1))
    holt_beta_range = DEMAND.get("holt_beta_range", (0.01, 0.5, 0.05))

    # ── MLflow run for Holt tuning ──
    if HAS_MLFLOW and branch_name:
        mlflow.start_run(run_name=f"holt_tuning_{branch_name}")
        mlflow.log_param("branch", branch_name)

    try:
        for alpha in np.arange(holt_alpha_range[0], holt_alpha_range[1], holt_alpha_range[2]):
            for beta in np.arange(holt_beta_range[0], holt_beta_range[1], holt_beta_range[2]):
                result = holts_exponential_smoothing(y, alpha=alpha, beta=beta, periods=periods)

                if HAS_MLFLOW and branch_name:
                    with mlflow.start_run(nested=True, run_name=f"alpha_{alpha:.2f}_beta_{beta:.2f}"):
                        mlflow.log_param("alpha", round(alpha, 2))
                        mlflow.log_param("beta", round(beta, 2))
                        mlflow.log_metric("mape", result["metrics"]["mape"])
                        mlflow.log_metric("mae", result["metrics"]["mae"])

                if result["metrics"]["mape"] < best_mape:
                    best_mape = result["metrics"]["mape"]
                    best_result = result
                    best_params = (round(alpha, 2), round(beta, 2))

        if HAS_MLFLOW and branch_name:
            mlflow.log_param("best_alpha", best_params[0])
            mlflow.log_param("best_beta", best_params[1])
            mlflow.log_metric("best_mape", best_mape)

    finally:
        if HAS_MLFLOW and branch_name:
            mlflow.end_run()

    return best_result, best_params


# ═══════════════════════════════════════════════════
# ML Model 3: Weighted Moving Average (baseline)
# ═══════════════════════════════════════════════════

def weighted_moving_average(y, periods=3):
    """
    Weighted moving average giving more weight to recent observations.
    Serves as a baseline to compare ML models against.
    """
    n = len(y)
    weights = np.arange(1, n + 1, dtype=float)
    weights = weights / weights.sum()
    wma = np.sum(y * weights)

    if n >= 2:
        recent_trend = (y[-1] - y[-2]) / y[-2] if y[-2] != 0 else 0
    else:
        recent_trend = 0

    forecast = []
    current = wma
    for h in range(periods):
        current = current * (1 + recent_trend * 0.5)  # dampened trend
        forecast.append(current)

    return {
        "forecast": forecast,
        "wma_value": wma,
        "trend_pct": recent_trend * 100,
    }


# ═══════════════════════════════════════════════════
# Ensemble: Combine all models
# ═══════════════════════════════════════════════════

def ensemble_forecast(poly_forecast, holt_forecast, wma_forecast,
                      poly_weight=None, holt_weight=None, wma_weight=None):
    """
    Weighted ensemble of all three models.
    Uses weights from DEMAND config if not provided.
    """
    if poly_weight is None:
        ensemble_weights = DEMAND.get("ensemble_weights", {"poly": 0.4, "holt": 0.4, "wma": 0.2})
        poly_weight = ensemble_weights.get("poly", 0.4)
        holt_weight = ensemble_weights.get("holt", 0.4)
        wma_weight = ensemble_weights.get("wma", 0.2)

    ensemble = []
    for i in range(len(poly_forecast)):
        p = poly_forecast[i] if i < len(poly_forecast) else poly_forecast[-1]
        h = holt_forecast[i] if i < len(holt_forecast) else holt_forecast[-1]
        w = wma_forecast[i] if i < len(wma_forecast) else wma_forecast[-1]
        ensemble.append(p * poly_weight + h * holt_weight + w * wma_weight)
    return ensemble


# ═══════════════════════════════════════════════════
# Main analysis pipeline
# ═══════════════════════════════════════════════════

def forecast_branch(branch_df, df150, df461, forecast_periods=3):
    """
    Run all forecasting models for a single branch.
    Includes MLflow tracking and LOO-CV metrics.
    """
    branch_name = branch_df["branch"].iloc[0]
    y = branch_df["sales"].values.astype(float)
    n = len(y)

    if n < 2:
        return {
            "branch": branch_name,
            "status": "insufficient_data",
            "message": f"Only {n} data points. Need at least 2.",
        }

    logger.info(f"Forecasting branch: {branch_name} (n={n} months)")

    # Feature engineering from all 3 sources
    X, feature_names, metadata = extract_features(branch_df, df150, df461)

    # ── Model 1: Polynomial Regression ──
    poly_result = fit_polynomial_regression(X, y, branch_name=branch_name)

    # Build future feature matrix
    future_t = np.arange(n, n + forecast_periods).astype(float)
    last_month = branch_df["month_num"].iloc[-1]
    future_months = np.array([((last_month + i) % 12) + 1 for i in range(forecast_periods)], dtype=float)
    future_month_sin = np.sin(2 * np.pi * future_months / 12)
    future_month_cos = np.cos(2 * np.pi * future_months / 12)

    small_n_threshold = DEMAND["small_n_threshold"]
    if n <= small_n_threshold:
        X_future = np.column_stack([future_t, future_month_sin])
    else:
        X_future = np.column_stack([
            future_t, future_t ** 2, future_month_sin, future_month_cos,
            np.full(forecast_periods, metadata["n_customers"]),
            np.full(forecast_periods, metadata["avg_staff_per_day"]),
        ])

    poly_forecast = predict_polynomial(X_future, poly_result["weights"])
    poly_forecast = np.maximum(poly_forecast, 0)

    # ── Model 2: Holt's Exponential Smoothing ──
    holt_result, holt_params = optimize_holt_params(y, branch_name=branch_name, periods=forecast_periods)

    # ── Model 3: Weighted Moving Average ──
    wma_result = weighted_moving_average(y, periods=forecast_periods)

    # ── Ensemble ──
    holt_clamped = [max(0, v) for v in holt_result["forecast"]]
    wma_clamped = [max(0, v) for v in wma_result["forecast"]]
    ens = ensemble_forecast(poly_forecast.tolist(), holt_clamped, wma_clamped)
    ens = [max(0, v) for v in ens]

    # Forecast labels
    forecast_labels = []
    last_year = branch_df["year"].iloc[-1]
    for i, m in enumerate(future_months):
        m_int = int(m)
        yr = last_year + (1 if (last_month + i) >= 12 else 0)
        forecast_labels.append(f"{MONTH_NAMES.get(m_int, 'Unknown')} {yr}")

    logger.info(f"  → Ensemble forecast: {[f'{v:,.0f}' for v in ens]}")
    poly_r2 = poly_result["metrics"]["r_squared"]
    holt_mape = holt_result["metrics"]["mape"]
    poly_mape = poly_result["metrics"]["mape"]
    logger.info(f"  → Poly R²={poly_r2:.2f}, Holt MAPE={holt_mape:.1f}%, Poly MAPE={poly_mape:.1f}%")
    if poly_result["cv_metrics"]:
        cv_mape = poly_result["cv_metrics"].get("cv_mape", 0)
        logger.info(f"  → Poly LOO-CV MAPE={cv_mape:.1f}%")

    return {
        "branch": branch_name,
        "status": "success",
        "data_points": n,
        "historical": {"months": branch_df["month"].tolist(), "sales": y.tolist()},
        "models": {
            "polynomial_regression": {
                "forecast": poly_forecast.tolist(),
                "metrics": {k: round(v, 2) for k, v in poly_result["metrics"].items()},
                "cv_metrics": {k: round(v, 2) for k, v in poly_result["cv_metrics"].items()},
                "best_lambda": round(poly_result["best_lambda"], 4),
                "feature_importance": dict(zip(
                    ["intercept"] + feature_names,
                    [round(w, 2) for w in poly_result["weights"]],
                )),
            },
            "exponential_smoothing": {
                "forecast": holt_clamped,
                "metrics": {k: round(v, 2) for k, v in holt_result["metrics"].items()},
                "optimal_params": {"alpha": holt_params[0], "beta": holt_params[1]},
            },
            "weighted_moving_avg": {
                "forecast": wma_clamped,
                "trend_pct": round(wma_result["trend_pct"], 1),
            },
        },
        "ensemble_forecast": ens,
        "forecast_labels": forecast_labels,
        "metadata": {k: round(v, 4) if isinstance(v, float) else v for k, v in metadata.items()},
    }


def run_full_demand_forecast(
    monthly_path="data/cleaned/cleaned_monthly_sales_(334).csv",
    orders_path="data/cleaned/customer_orders_(150).csv",
    attendance_path="data/cleaned/cleaned_attendance_(461).csv",
    forecast_periods=3,
):
    """
    Complete demand forecasting pipeline for all branches.
    Includes MLflow tracking and comprehensive logging.
    """
    logger.info("=" * 60)
    logger.info("DEMAND FORECASTING ANALYSIS")
    logger.info("=" * 60)

    logger.info("\n[1/4] Loading data...")
    df334, df150, df461 = load_data(monthly_path, orders_path, attendance_path)

    logger.info("\n[2/4] Preparing time series...")
    branch_series = prepare_time_series(df334)
    for branch, bdf in branch_series.items():
        logger.info(f"  → {branch}: {len(bdf)} months of data")

    logger.info("\n[3/4] Running forecasting models per branch...")
    results = {}
    for branch, bdf in branch_series.items():
        result = forecast_branch(bdf, df150, df461, forecast_periods=forecast_periods)
        results[branch] = result
        if result["status"] == "success":
            ens = result["ensemble_forecast"]
            logger.info(f"  → {branch}: Ensemble = {[f'{v:,.0f}' for v in ens]}")
            poly_r2 = result["models"]["polynomial_regression"]["metrics"]["r_squared"]
            holt_mape = result["models"]["exponential_smoothing"]["metrics"]["mape"]
            logger.info(f"    Poly R²={poly_r2:.2f}, Holt MAPE={holt_mape:.1f}%")
            meta = result["metadata"]
            logger.info(f"    Customers(150): {meta['n_customers']}, Staff/day(461): {meta['avg_staff_per_day']}")
        else:
            logger.info(f"  → {branch}: {result['message']}")

    logger.info("\n[4/4] Generating summary...")
    summary = _build_summary(results)
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)

    return {"branch_forecasts": results, "summary": summary}


def _build_summary(results):
    """Build a text summary of the demand forecasting results."""
    lines = ["DEMAND FORECASTING SUMMARY", "=" * 40]
    for branch, r in results.items():
        lines.append(f"\n--- {branch.upper()} ---")
        if r["status"] != "success":
            lines.append(f"  {r['message']}")
            continue
        lines.append(f"  Data: {r['data_points']} months")
        poly_mape = r["models"]["polynomial_regression"]["metrics"]["mape"]
        holt_mape = r["models"]["exponential_smoothing"]["metrics"]["mape"]
        best = "Polynomial Regression" if poly_mape < holt_mape else "Exponential Smoothing"
        lines.append(f"  Best Model: {best} (MAPE: {min(poly_mape, holt_mape):.1f}%)")
        for label, val in zip(r["forecast_labels"], r["ensemble_forecast"]):
            lines.append(f"  Forecast {label}: {val:,.0f} LBP")
        meta = r["metadata"]
        lines.append(f"  Customers (150): {meta['n_customers']}, Repeat Rate: {meta['repeat_customer_rate']:.1%}")
        lines.append(f"  Avg Staff/Day (461): {meta['avg_staff_per_day']:.1f}, Avg Hours: {meta['avg_work_hours']:.1f}h")
        hist = r["historical"]["sales"]
        if len(hist) >= 2 and hist[0] > 0:
            growth = (hist[-1] - hist[0]) / hist[0] * 100
            lines.append(f"  Historical Growth: {growth:+.1f}%")
    return "\n".join(lines)


# === Agent-callable function for OpenClaw integration ===

def get_demand_forecast(
    branch=None,
    periods=3,
    monthly_path="data/cleaned/cleaned_monthly_sales_(334).csv",
    orders_path="data/cleaned/customer_orders_(150).csv",
    attendance_path="data/cleaned/cleaned_attendance_(461).csv",
):
    """
    OpenClaw-compatible function: Returns demand forecast per branch.
    Uses ensemble of Polynomial Regression, Holt's Exponential Smoothing,
    and Weighted Moving Average. Features from customer orders (150)
    and staff attendance (461).

    Function signature and return format MUST NOT change.
    """
    try:
        df334, df150, df461 = load_data(monthly_path, orders_path, attendance_path)
        branch_series = prepare_time_series(df334)

        if branch:
            if branch not in branch_series:
                return {"status": "error", "message": f"Branch '{branch}' not found. Available: {list(branch_series.keys())}"}
            branch_series = {branch: branch_series[branch]}

        result = {"status": "success", "forecasts": []}

        for b_name, bdf in branch_series.items():
            forecast = forecast_branch(bdf, df150, df461, forecast_periods=periods)

            if forecast["status"] != "success":
                result["forecasts"].append({
                    "branch": b_name, "status": "error", "message": forecast["message"],
                })
                continue

            poly_mape = forecast["models"]["polynomial_regression"]["metrics"]["mape"]
            holt_mape = forecast["models"]["exponential_smoothing"]["metrics"]["mape"]
            best_model = "polynomial_regression" if poly_mape < holt_mape else "exponential_smoothing"

            result["forecasts"].append({
                "branch": b_name,
                "status": "success",
                "data_points": forecast["data_points"],
                "ensemble_forecast": [
                    {"period": label, "value": round(val, 0)}
                    for label, val in zip(forecast["forecast_labels"], forecast["ensemble_forecast"])
                ],
                "best_model": best_model.replace("_", " ").title(),
                "best_model_mape": round(min(poly_mape, holt_mape), 1),
                "poly_r_squared": forecast["models"]["polynomial_regression"]["metrics"]["r_squared"],
                "historical_sales": forecast["historical"]["sales"],
                "historical_months": forecast["historical"]["months"],
                "customer_insights": {
                    "total_customers": forecast["metadata"]["n_customers"],
                    "avg_spend": round(forecast["metadata"]["avg_spend_per_customer"], 0),
                    "repeat_rate": round(forecast["metadata"]["repeat_customer_rate"] * 100, 1),
                    "total_orders": forecast["metadata"]["total_orders"],
                },
                "staffing_insights": {
                    "avg_staff_per_day": round(forecast["metadata"]["avg_staff_per_day"], 1),
                    "avg_work_hours": round(forecast["metadata"]["avg_work_hours"], 1),
                    "working_days_recorded": forecast["metadata"]["working_days_recorded"],
                },
            })

        return result

    except Exception as e:
        logger.error(f"Error in get_demand_forecast: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}
