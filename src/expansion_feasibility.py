"""
Module 3: Expansion Feasibility
Evaluates whether opening a new Conut branch is feasible and recommends
candidate location profiles based on existing branch performance data.

Data inputs:
- cleaned_monthly_sales_(334).csv: Monthly sales by branch — revenue trends
- cleaned_avg_sales_by_menu_(435).csv: Channel mix, customer counts, avg spend
- customer_orders_(150).csv: Customer loyalty (repeat rate, order frequency)
- cleaned_attendance_(461).csv: Staffing levels and labor hours
- cleaned_tax_report_(194).csv: Tax revenue (VAT) by branch — profitability proxy

ML Model: K-Means Clustering (sklearn with numpy fallback) to segment branches by performance,
plus a Weighted Scoring Model to rank expansion feasibility.

Key outputs:
- Branch performance profiles (revenue, growth, efficiency, loyalty)
- Cluster analysis of branch types
- Expansion feasibility score and recommendation
- Ideal new branch profile based on top-performing branches
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from config import validate_dataframe, get_logger, RANDOM_SEED, EXPANSION, MONTH_ORDER

# Set random seed at module level for reproducibility
np.random.seed(RANDOM_SEED)

logger = get_logger(__name__)

# Try to import sklearn components
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, will use pure numpy KMeans")

try:
    from sklearn.metrics import silhouette_score
    SILHOUETTE_AVAILABLE = True
except ImportError:
    SILHOUETTE_AVAILABLE = False
    logger.warning("silhouette_score not available, will default to k=2")

# Try to import MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available, logging disabled")


def parse_duration_hours(s):
    """Convert HH:MM:SS to decimal hours."""
    try:
        parts = str(s).split(":")
        return int(parts[0]) + int(parts[1]) / 60 + int(parts[2]) / 3600
    except Exception:
        return 0.0


def load_data(
    monthly_path="data/cleaned/cleaned_monthly_sales_(334).csv",
    menu_path="data/cleaned/cleaned_avg_sales_by_menu_(435).csv",
    orders_path="data/cleaned/customer_orders_(150).csv",
    attendance_path="data/cleaned/cleaned_attendance_(461).csv",
    tax_path="data/cleaned/cleaned_tax_report_(194).csv",
):
    """Load all five data sources."""
    df334 = pd.read_csv(monthly_path)
    df435 = pd.read_csv(menu_path)
    df150 = pd.read_csv(orders_path)
    df461 = pd.read_csv(attendance_path)
    df194 = pd.read_csv(tax_path)
    
    logger.info(f"Loaded 334: {len(df334)} rows (monthly sales)")
    logger.info(f"Loaded 435: {len(df435)} rows (avg sales by menu)")
    logger.info(f"Loaded 150: {len(df150)} rows (customer orders)")
    logger.info(f"Loaded 461: {len(df461)} rows (attendance)")
    logger.info(f"Loaded 194: {len(df194)} rows (tax report)")
    
    return df334, df435, df150, df461, df194


def build_branch_profiles(df334, df435, df150, df461, df194):
    """
    Build a comprehensive performance profile for each branch using all 5 sources.

    Returns a DataFrame with one row per branch and multiple performance features.
    """
    branches = sorted(df334["branch"].unique())
    profiles = []

    for branch in branches:
        profile = {"branch": branch}

        # ── 334: Revenue metrics ──
        b334 = df334[df334["branch"] == branch].copy()
        b334["month_num"] = b334["month"].map(MONTH_ORDER)
        b334 = b334.sort_values("month_num")

        sales_values = b334["sales"].values
        profile["total_revenue"] = float(sales_values.sum())
        profile["avg_monthly_revenue"] = float(sales_values.mean())
        profile["revenue_std"] = float(sales_values.std()) if len(sales_values) > 1 else 0
        profile["n_months"] = len(sales_values)

        # Growth rate (first to last month)
        if len(sales_values) >= 2 and sales_values[0] > 0:
            profile["growth_rate"] = float((sales_values[-1] / sales_values[0]) - 1)
        else:
            profile["growth_rate"] = 0.0

        # Revenue momentum (avg of last 2 months vs first 2 months)
        if len(sales_values) >= 4:
            early = sales_values[:2].mean()
            late = sales_values[-2:].mean()
            profile["momentum"] = float((late / early) - 1) if early > 0 else 0.0
        else:
            profile["momentum"] = profile["growth_rate"]

        # Revenue consistency (coefficient of variation — lower = more consistent)
        profile["revenue_cv"] = float(profile["revenue_std"] / profile["avg_monthly_revenue"]) if profile["avg_monthly_revenue"] > 0 else 1.0

        # ── 435: Channel & customer metrics ──
        b435 = df435[df435["branch"] == branch]
        active_channels = b435[b435["sales"] > 0]
        profile["n_channels"] = len(active_channels)
        profile["total_customers_435"] = float(b435["total_customers"].max()) if len(b435) > 0 else 0
        profile["avg_customer_spend"] = float(b435["total_avg_customer"].max()) if len(b435) > 0 else 0
        profile["total_sales_435"] = float(b435["total_sales"].max()) if len(b435) > 0 else 0

        # Channel diversity (how spread out revenue is across channels)
        if len(active_channels) > 0:
            channel_shares = active_channels["sales"].values / active_channels["sales"].sum()
            entropy = -np.sum(channel_shares * np.log(channel_shares + 1e-10))
            max_entropy = np.log(len(active_channels)) if len(active_channels) > 1 else 1
            profile["channel_diversity"] = float(entropy / max_entropy) if max_entropy > 0 else 0
        else:
            profile["channel_diversity"] = 0

        # ── 150: Customer loyalty metrics ──
        b150 = df150[df150["branch"] == branch]
        profile["n_customers_150"] = len(b150)
        if len(b150) > 0:
            profile["repeat_rate"] = float((b150["order_count"] > 1).sum() / len(b150))
            profile["avg_order_count"] = float(b150["order_count"].mean())
            profile["avg_total_spent"] = float(b150["total_spent"].mean())
        else:
            profile["repeat_rate"] = 0
            profile["avg_order_count"] = 0
            profile["avg_total_spent"] = 0

        # ── 461: Staffing efficiency ──
        b461 = df461[df461["branch"] == branch].copy()
        if len(b461) > 0:
            b461["date_in"] = pd.to_datetime(b461["date_in"])
            b461["hours"] = b461["work_duration"].apply(parse_duration_hours)
            days = b461["date_in"].dt.date.nunique()
            profile["avg_staff_per_day"] = float(len(b461) / days) if days > 0 else 0
            profile["avg_work_hours"] = float(b461["hours"].mean())
            profile["total_labor_hours"] = float(b461["hours"].sum())

            dec_rev = df334[(df334["branch"] == branch) & (df334["month"] == "December")]["sales"].sum()
            profile["revenue_per_labor_hour"] = float(dec_rev / profile["total_labor_hours"]) if profile["total_labor_hours"] > 0 else 0
        else:
            profile["avg_staff_per_day"] = 0
            profile["avg_work_hours"] = 0
            profile["total_labor_hours"] = 0
            profile["revenue_per_labor_hour"] = 0

        # ── 194: Tax / profitability proxy ──
        b194 = df194[df194["branch"] == branch]
        profile["vat_total"] = float(b194["vat_11_total"].values[0]) if len(b194) > 0 else 0
        profile["tax_ratio"] = float(profile["vat_total"] / profile["total_revenue"]) if profile["total_revenue"] > 0 else 0

        profiles.append(profile)

    return pd.DataFrame(profiles)


def normalize_features(X):
    """Min-max normalize to [0, 1]."""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    return (X - X_min) / X_range


def kmeans_cluster_sklearn(X, k=2):
    """K-Means clustering using sklearn with k-means++ initialization."""
    try:
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            random_state=RANDOM_SEED,
            n_init=10
        )
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        logger.info(f"sklearn KMeans with k={k} completed successfully")
        return labels, centroids, True
    except Exception as e:
        logger.error(f"sklearn KMeans failed: {e}")
        return None, None, False


def kmeans_cluster_numpy(X, k=2, max_iter=100):
    """Pure numpy K-Means clustering (fallback)."""
    n = len(X)
    if n <= k:
        return np.arange(n), X.copy()

    idx = np.argsort(X[:, 0])
    step = n // k
    centroids = X[idx[::step]][:k].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        distances = np.array([np.sqrt(((X - c) ** 2).sum(axis=1)) for c in centroids])
        new_labels = distances.argmin(axis=0)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                centroids[j] = X[mask].mean(axis=0)

    logger.info(f"numpy KMeans with k={k} completed")
    return labels, centroids


def choose_optimal_k(X):
    """Choose optimal k (2 or 3) using silhouette score."""
    if not SILHOUETTE_AVAILABLE:
        logger.info("silhouette_score not available, defaulting to k=2")
        return 2

    try:
        # Try k=2 and k=3
        scores = {}
        for k in [2, 3]:
            if SKLEARN_AVAILABLE:
                labels, _, success = kmeans_cluster_sklearn(X, k=k)
                if not success:
                    labels, _ = kmeans_cluster_numpy(X, k=k)
            else:
                labels, _ = kmeans_cluster_numpy(X, k=k)

            score = silhouette_score(X, labels)
            scores[k] = score
            logger.info(f"Silhouette score for k={k}: {score:.4f}")

        best_k = max(scores, key=scores.get)
        logger.info(f"Optimal k selected: {best_k} (score: {scores[best_k]:.4f})")
        return best_k
    except Exception as e:
        logger.error(f"Error choosing optimal k: {e}. Defaulting to k=2")
        return 2


def kmeans_cluster(X, k=2):
    """K-Means clustering with sklearn fallback to numpy."""
    if SKLEARN_AVAILABLE:
        labels, centroids, success = kmeans_cluster_sklearn(X, k=k)
        if success:
            return labels, centroids
        else:
            logger.info("Falling back to numpy KMeans")
            return kmeans_cluster_numpy(X, k=k)
    else:
        return kmeans_cluster_numpy(X, k=k)


def calculate_expansion_score(profiles_df):
    """
    Calculate a weighted expansion feasibility score based on the overall
    health of the Conut network.

    Returns: score (0-100), dimension scores, recommendation
    """
    df = profiles_df.copy()
    dimensions = {}

    # 1. Revenue growth
    growth_rates = df["growth_rate"].values
    positive_growth = growth_rates[growth_rates > -0.5]
    avg_growth = positive_growth.mean() if len(positive_growth) > 0 else 0
    dimensions["revenue_growth"] = min(max(avg_growth * 30 + 50, 0), 100)

    # 2. Revenue scale
    total_rev = df["total_revenue"].sum()
    dimensions["revenue_scale"] = min(max((total_rev / 2e10) * 90, 10), 100)

    # 3. Customer base
    total_cust = df["total_customers_435"].sum()
    dimensions["customer_base"] = min(max((total_cust / 15000) * 80, 10), 100)

    # 4. Operational efficiency
    eff_branches = df[df["revenue_per_labor_hour"] > 0]
    if len(eff_branches) > 0:
        avg_eff = eff_branches["revenue_per_labor_hour"].mean()
        dimensions["operational_efficiency"] = min(max((avg_eff / 15e6) * 70, 10), 100)
    else:
        dimensions["operational_efficiency"] = 50

    # 5. Channel diversification
    avg_channels = df["n_channels"].mean()
    avg_diversity = df["channel_diversity"].mean()
    dimensions["channel_diversification"] = min(max(avg_channels * 20 + avg_diversity * 30, 10), 100)

    # 6. Customer loyalty
    loyalty_branches = df[df["repeat_rate"] > 0]
    if len(loyalty_branches) > 0:
        avg_loyalty = loyalty_branches["repeat_rate"].mean()
        dimensions["customer_loyalty"] = min(max(avg_loyalty * 300, 10), 100)
    else:
        dimensions["customer_loyalty"] = 30

    # 7. Tax health
    avg_tax = df["tax_ratio"].mean()
    dimensions["tax_health"] = min(max(avg_tax * 500, 10), 100)

    weights = EXPANSION["score_weights"]

    total_score = sum(dimensions[k] * weights[k] for k in weights)

    if total_score >= 75:
        recommendation = "Strongly Recommended"
        reasoning = "The network shows strong growth, healthy revenue, and operational maturity. Expansion risk is low."
    elif total_score >= 55:
        recommendation = "Recommended with Caution"
        reasoning = "The network is growing but has some gaps (e.g., inconsistent branches, limited channels in some locations). Address weaknesses before expanding."
    elif total_score >= 40:
        recommendation = "Conditional"
        reasoning = "Mixed signals — some branches are thriving while others struggle. Focus on stabilizing underperformers before opening a new location."
    else:
        recommendation = "Not Recommended"
        reasoning = "The current network needs significant improvement before expansion. Revenue trends and operational metrics suggest consolidation is the priority."

    return {
        "total_score": round(total_score, 1),
        "dimensions": {k: round(v, 1) for k, v in dimensions.items()},
        "weights": weights,
        "recommendation": recommendation,
        "reasoning": reasoning,
    }


def generate_ideal_branch_profile(profiles_df):
    """
    Based on top-performing branches, generate an ideal profile for a new location.
    """
    df = profiles_df.copy()

    best_idx = df["total_revenue"].idxmax()
    best_branch = df.loc[best_idx, "branch"]

    ideal = {
        "benchmark_branch": best_branch,
        "target_monthly_revenue": round(df["avg_monthly_revenue"].max(), 0),
        "target_customers": int(df["total_customers_435"].max()),
        "target_avg_spend": round(df["avg_customer_spend"].max(), 0),
        "recommended_channels": int(df["n_channels"].max()),
        "target_repeat_rate": round(df["repeat_rate"].max() * 100, 1),
        "recommended_staff_per_day": round(df[df["avg_staff_per_day"] > 0]["avg_staff_per_day"].mean(), 1) if (df["avg_staff_per_day"] > 0).any() else 3,
        "target_avg_work_hours": round(df[df["avg_work_hours"] > 0]["avg_work_hours"].mean(), 1) if (df["avg_work_hours"] > 0).any() else 8,
        "location_criteria": [],
    }

    if df["n_channels"].max() >= 2:
        ideal["location_criteria"].append("Support delivery + dine-in to maximize channel revenue")
    if df[df["total_customers_435"] > 3000].shape[0] >= 2:
        ideal["location_criteria"].append("Target high-traffic area (3000+ potential customers)")
    if (df["growth_rate"] > 1).any():
        ideal["location_criteria"].append("Newer locations show explosive growth — consider underserved areas")
    if df["avg_customer_spend"].mean() > 1e6:
        ideal["location_criteria"].append(f"Average customer spend across network is {df['avg_customer_spend'].mean():,.0f} LBP — price menu accordingly")

    return ideal


def run_full_expansion_analysis(
    monthly_path="data/cleaned/cleaned_monthly_sales_(334).csv",
    menu_path="data/cleaned/cleaned_avg_sales_by_menu_(435).csv",
    orders_path="data/cleaned/customer_orders_(150).csv",
    attendance_path="data/cleaned/cleaned_attendance_(461).csv",
    tax_path="data/cleaned/cleaned_tax_report_(194).csv",
):
    """Full pipeline: load → profile → cluster → score → recommend."""
    logger.info("=" * 60)
    logger.info("EXPANSION FEASIBILITY ANALYSIS")
    logger.info("=" * 60)

    logger.info("[1/5] Loading data...")
    df334, df435, df150, df461, df194 = load_data(
        monthly_path, menu_path, orders_path, attendance_path, tax_path
    )

    logger.info("[2/5] Building branch performance profiles...")
    profiles = build_branch_profiles(df334, df435, df150, df461, df194)
    logger.info(f"  → {len(profiles)} branch profiles built")
    for _, row in profiles.iterrows():
        logger.info(f"  → {row['branch']}: revenue={row['total_revenue']:,.0f}, growth={row['growth_rate']*100:.1f}%, customers={row['total_customers_435']:.0f}")

    logger.info("[3/5] Clustering branches by performance...")
    cluster_features = [
        "avg_monthly_revenue", "growth_rate", "revenue_cv",
        "total_customers_435", "avg_customer_spend", "repeat_rate",
    ]
    X = profiles[cluster_features].fillna(0).values.astype(float)
    X_norm = normalize_features(X)
    
    # Choose optimal k using silhouette score
    optimal_k = choose_optimal_k(X_norm)
    labels, centroids = kmeans_cluster(X_norm, k=optimal_k)
    profiles["cluster"] = labels

    cluster_means = profiles.groupby("cluster")["avg_monthly_revenue"].mean()
    high_cluster = cluster_means.idxmax()
    profiles["cluster_label"] = profiles["cluster"].apply(
        lambda x: "High Performer" if x == high_cluster else "Growing / Stabilizing"
    )
    for _, row in profiles.iterrows():
        logger.info(f"  → {row['branch']}: {row['cluster_label']}")

    logger.info("[4/5] Calculating expansion feasibility score...")
    score_result = calculate_expansion_score(profiles)
    logger.info(f"  → Total Score: {score_result['total_score']}/100")
    logger.info(f"  → Recommendation: {score_result['recommendation']}")
    for dim, val in score_result["dimensions"].items():
        logger.info(f"     {dim}: {val}/100")

    logger.info("[5/5] Generating ideal branch profile...")
    ideal = generate_ideal_branch_profile(profiles)
    logger.info(f"  → Benchmark: {ideal['benchmark_branch']}")
    logger.info(f"  → Target monthly revenue: {ideal['target_monthly_revenue']:,.0f} LBP")

    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)

    # MLflow logging
    if MLFLOW_AVAILABLE:
        try:
            mlflow.start_run()
            
            # Log cluster assignments
            for _, row in profiles.iterrows():
                mlflow.log_param(f"cluster_{row['branch']}", row["cluster_label"])
            
            # Log silhouette score and optimal k
            if SILHOUETTE_AVAILABLE:
                try:
                    from sklearn.metrics import silhouette_score
                    sil_score = silhouette_score(X_norm, labels)
                    mlflow.log_metric("silhouette_score", sil_score)
                    logger.info(f"MLflow logged silhouette_score: {sil_score:.4f}")
                except Exception as e:
                    logger.warning(f"Could not log silhouette score: {e}")
            
            mlflow.log_param("optimal_k", optimal_k)
            mlflow.log_param("n_branches", len(profiles))
            
            # Log expansion score dimensions
            for dim, val in score_result["dimensions"].items():
                mlflow.log_metric(f"expansion_score_{dim}", val)
            
            mlflow.log_metric("expansion_score_total", score_result["total_score"])
            
            mlflow.end_run()
            logger.info("MLflow logging completed")
        except Exception as e:
            logger.error(f"MLflow logging error: {e}")

    return {
        "status": "success",
        "branch_profiles": profiles.to_dict("records"),
        "clustering": {
            "features_used": cluster_features,
            "n_clusters": optimal_k,
            "assignments": {
                row["branch"]: row["cluster_label"]
                for _, row in profiles.iterrows()
            },
        },
        "expansion_score": score_result,
        "ideal_branch_profile": ideal,
    }


def get_expansion_assessment(
    monthly_path="data/cleaned/cleaned_monthly_sales_(334).csv",
    menu_path="data/cleaned/cleaned_avg_sales_by_menu_(435).csv",
    orders_path="data/cleaned/customer_orders_(150).csv",
    attendance_path="data/cleaned/cleaned_attendance_(461).csv",
    tax_path="data/cleaned/cleaned_tax_report_(194).csv",
):
    """
    OpenClaw-compatible function: Get expansion feasibility assessment.
    """
    try:
        result = run_full_expansion_analysis(
            monthly_path, menu_path, orders_path, attendance_path, tax_path
        )

        def clean(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean(i) for i in obj]
            return obj

        return clean(result)
    except Exception as e:
        logger.error(f"Error in get_expansion_assessment: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    result = run_full_expansion_analysis()
    if result["status"] == "success":
        logger.info(f"\n\nFINAL SCORE: {result['expansion_score']['total_score']}/100")
        logger.info(f"RECOMMENDATION: {result['expansion_score']['recommendation']}")
