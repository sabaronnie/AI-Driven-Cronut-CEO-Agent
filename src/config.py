"""
Centralized configuration for the Conut Chief of Operations system.
All tunable constants and file paths in one place.
"""

import os
import logging

# ── Project paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
MLFLOW_DIR = os.path.join(PROJECT_ROOT, "mlruns")
METRICS_DIR = os.path.join(PROJECT_ROOT, "metrics")

# ── Data file paths ──
FILES = {
    "191": os.path.join(DATA_DIR, "cleaned_sales_by_item_(191).csv"),
    "502": os.path.join(DATA_DIR, "sales_detail_(502).csv"),
    "136": os.path.join(DATA_DIR, "cleaned_revenue_summary_(136).csv"),
    "334": os.path.join(DATA_DIR, "cleaned_monthly_sales_(334).csv"),
    "435": os.path.join(DATA_DIR, "cleaned_avg_sales_by_menu_(435).csv"),
    "461": os.path.join(DATA_DIR, "cleaned_attendance_(461).csv"),
    "150": os.path.join(DATA_DIR, "customer_orders_(150).csv"),
    "194": os.path.join(DATA_DIR, "cleaned_tax_report_(194).csv"),
}

# ── Random seed for reproducibility ──
RANDOM_SEED = 42

# ── Model 1: Combo Optimization ──
COMBO = {
    "min_support": 0.05,
    "min_confidence": 0.3,
    "min_lift": 1.0,
    "top_n": 10,
    "exclude_items": {"DELIVERY CHARGE", "WATER"},
    "modifier_suffixes": (".", "(P)", "(R)"),
    "exclude_categories": {"sauces", "spreads"},
}

# ── Model 2: Demand Forecasting ──
DEMAND = {
    "ridge_lambda": 0.1,
    "ridge_lambda_candidates": [0.01, 0.05, 0.1, 0.5, 1.0],
    "poly_degree": 2,
    "holt_alpha_range": (0.1, 0.9, 0.1),   # start, stop, step
    "holt_beta_range": (0.01, 0.5, 0.05),
    "ensemble_weights": {"poly": 0.4, "holt": 0.4, "wma": 0.2},
    "wma_weights": [0.1, 0.2, 0.3, 0.4],
    "max_features_small_n": 2,
    "small_n_threshold": 5,
}

# ── Model 3: Expansion Feasibility ──
EXPANSION = {
    "n_clusters": 2,
    "kmeans_max_iter": 100,
    "score_weights": {
        "revenue_growth": 0.25,
        "revenue_scale": 0.20,
        "customer_base": 0.15,
        "operational_efficiency": 0.15,
        "channel_diversification": 0.10,
        "customer_loyalty": 0.10,
        "tax_health": 0.05,
    },
}

# ── Model 4: Shift Staffing ──
STAFFING = {
    "ridge_lambda": 0.5,
    "ridge_lambda_candidates": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "shift_bins": {"Morning": (5, 12), "Afternoon": (12, 17), "Evening": (17, 24)},
    "cv_folds": 5,
}

# ── Model 6: Branch Location Recommender ──
LOCATION = {
    "demand_weights": {
        "population": 0.35,
        "social_activity": 0.25,
        "traffic_index": 0.20,
        "university_presence": 0.10,
        "tourism_score": 0.10,
    },
    "ridge_lambda_candidates": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "poly_degree": 2,
    "score_weights": {
        "gap_score": 0.40,
        "demand_score": 0.25,
        "affordability": 0.15,
        "growth_potential": 0.20,
    },
}

# ── Model 5: Coffee & Milkshake Growth ──
COFFEE = {
    "coffee_divisions": ["Hot-Coffee Based", "Frappes"],
    "milkshake_divisions": ["Shakes"],
    "exclude_cross_sell": {"WATER", "DELIVERY CHARGE"},
    "delivery_potential_pct": 0.05,
}

# ── Month mapping ──
MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
MONTH_NAMES = {v: k for k, v in MONTH_ORDER.items()}


# ── Logging setup ──
def get_logger(name):
    """Get a configured logger for a module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def validate_dataframe(df, required_columns, source_name):
    """Validate that a DataFrame has the expected columns."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Data source '{source_name}' is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return True
