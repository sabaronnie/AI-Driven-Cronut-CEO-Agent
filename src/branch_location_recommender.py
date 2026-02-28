"""
Module 6: Branch Location Recommender
Analyzes Lebanese areas to recommend optimal locations for new Conut branches
based on market gap analysis: population, social activity, traffic, and
competitor density (coffee shops + sweet shops/bakeries).

Methodology:
1. Build a dataset of Lebanese areas with demographic and competitor data
2. Compute a "demand proxy" from population, social activity, and traffic
3. Compute competitor saturation (competitors per demand unit)
4. Find the market gap: expected competitors vs actual competitors
5. Rank areas by gap score (higher gap = more opportunity)
6. Use Ridge Regression to model the non-linear relationship between
   demand factors and competitor counts, then predict expected competitors

Data sources:
- Population estimates from CAS 2018-2019 survey, World Population Review,
  and citypopulation.de
- Competitor counts estimated from Yelleb.com directory, Wanderlog, BAM Lebanon,
  TripAdvisor, and The961 listings
- Traffic indices estimated from World Bank transport studies, Tari2ak data,
  and MATEC conference papers on Lebanese traffic congestion
- Social activity indices from commercial density reports and nightlife/tourism data

Existing Conut branches (EXCLUDED from recommendations):
- Conut - Tyre (Tyre)
- Conut Jnah (Beirut - Jnah)
- Main Street Coffee (Batroun)
"""

import numpy as np
import pandas as pd
from config import get_logger, RANDOM_SEED

np.random.seed(RANDOM_SEED)
logger = get_logger(__name__)

# Try sklearn imports with fallback
try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available — using numpy fallback for regression")

# Try mlflow
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


# ══════════════════════════════════════════════════════════════════════════════
# LEBANESE AREA DATASET
# ══════════════════════════════════════════════════════════════════════════════
# Each area has:
#   - population: estimated resident population (sources: CAS 2018-19, WPR)
#   - social_activity: index 0-100 representing nightlife, events, foot traffic
#     (derived from tourism reports, commercial density, university presence)
#   - traffic_index: index 0-100 representing daily vehicle/pedestrian flow
#     (derived from World Bank transport data, Tari2ak, congestion studies)
#   - coffee_shops: estimated number of coffee shops/cafes in the area
#     (from Yelleb, Wanderlog, TripAdvisor, Google Maps cross-reference)
#   - sweet_bakery_shops: estimated sweet shops, patisseries, bakeries, donut shops
#     (from Yelleb, BAM Lebanon, The961 listings)
#   - has_conut: whether Conut already operates here (1=yes, 0=no)
#   - university_presence: 0-3 scale (0=none, 1=small, 2=medium, 3=major hub)
#   - tourism_score: 0-10 representing tourist/visitor attraction level
#   - avg_rent_index: 0-100 representing commercial rent cost (higher = pricier)
#
# IMPORTANT: These are best-available estimates compiled from multiple public
# sources. Exact figures are not available due to Lebanon's lack of census data.
# ══════════════════════════════════════════════════════════════════════════════

AREA_DATA = [
    # ── BEIRUT NEIGHBORHOODS ──
    {
        "area": "Beirut - Hamra",
        "governorate": "Beirut",
        "population": 45000,
        "social_activity": 92,
        "traffic_index": 88,
        "coffee_shops": 45,
        "sweet_bakery_shops": 30,
        "has_conut": 0,
        "university_presence": 3,  # AUB, LAU nearby
        "tourism_score": 8,
        "avg_rent_index": 85,
    },
    {
        "area": "Beirut - Achrafieh",
        "governorate": "Beirut",
        "population": 55000,
        "social_activity": 90,
        "traffic_index": 85,
        "coffee_shops": 50,
        "sweet_bakery_shops": 35,
        "has_conut": 0,
        "university_presence": 2,  # USJ, ALBA
        "tourism_score": 7,
        "avg_rent_index": 90,
    },
    {
        "area": "Beirut - Verdun",
        "governorate": "Beirut",
        "population": 35000,
        "social_activity": 80,
        "traffic_index": 82,
        "coffee_shops": 30,
        "sweet_bakery_shops": 20,
        "has_conut": 0,
        "university_presence": 1,
        "tourism_score": 6,
        "avg_rent_index": 88,
    },
    {
        "area": "Beirut - Mar Mikhael",
        "governorate": "Beirut",
        "population": 15000,
        "social_activity": 95,
        "traffic_index": 78,
        "coffee_shops": 35,
        "sweet_bakery_shops": 15,
        "has_conut": 0,
        "university_presence": 1,
        "tourism_score": 9,
        "avg_rent_index": 75,
    },
    {
        "area": "Beirut - Downtown/BCD",
        "governorate": "Beirut",
        "population": 8000,
        "social_activity": 70,
        "traffic_index": 90,
        "coffee_shops": 20,
        "sweet_bakery_shops": 10,
        "has_conut": 0,
        "university_presence": 0,
        "tourism_score": 8,
        "avg_rent_index": 95,
    },
    {
        "area": "Beirut - Jnah",
        "governorate": "Beirut",
        "population": 40000,
        "social_activity": 55,
        "traffic_index": 75,
        "coffee_shops": 15,
        "sweet_bakery_shops": 12,
        "has_conut": 1,  # EXCLUDED — Conut Jnah
        "university_presence": 1,
        "tourism_score": 3,
        "avg_rent_index": 50,
    },
    {
        "area": "Beirut - Badaro",
        "governorate": "Beirut",
        "population": 20000,
        "social_activity": 75,
        "traffic_index": 70,
        "coffee_shops": 18,
        "sweet_bakery_shops": 10,
        "has_conut": 0,
        "university_presence": 1,
        "tourism_score": 6,
        "avg_rent_index": 65,
    },
    {
        "area": "Beirut - Geitawi/Rmeil",
        "governorate": "Beirut",
        "population": 25000,
        "social_activity": 72,
        "traffic_index": 65,
        "coffee_shops": 15,
        "sweet_bakery_shops": 8,
        "has_conut": 0,
        "university_presence": 1,
        "tourism_score": 5,
        "avg_rent_index": 60,
    },

    # ── MOUNT LEBANON ──
    {
        "area": "Jounieh",
        "governorate": "Keserwan-Jbeil",
        "population": 120000,
        "social_activity": 78,
        "traffic_index": 80,
        "coffee_shops": 35,
        "sweet_bakery_shops": 25,
        "has_conut": 0,
        "university_presence": 2,  # NDU, USEK
        "tourism_score": 8,
        "avg_rent_index": 70,
    },
    {
        "area": "Byblos (Jbeil)",
        "governorate": "Keserwan-Jbeil",
        "population": 40000,
        "social_activity": 72,
        "traffic_index": 55,
        "coffee_shops": 20,
        "sweet_bakery_shops": 15,
        "has_conut": 0,
        "university_presence": 1,  # LAU Byblos
        "tourism_score": 9,
        "avg_rent_index": 65,
    },
    {
        "area": "Kaslik/Zouk",
        "governorate": "Keserwan-Jbeil",
        "population": 50000,
        "social_activity": 82,
        "traffic_index": 75,
        "coffee_shops": 28,
        "sweet_bakery_shops": 18,
        "has_conut": 0,
        "university_presence": 2,  # USEK
        "tourism_score": 6,
        "avg_rent_index": 72,
    },
    {
        "area": "Dbayeh/Antelias",
        "governorate": "Metn",
        "population": 65000,
        "social_activity": 70,
        "traffic_index": 82,
        "coffee_shops": 22,
        "sweet_bakery_shops": 15,
        "has_conut": 0,
        "university_presence": 1,
        "tourism_score": 4,
        "avg_rent_index": 75,
    },
    {
        "area": "Broummana",
        "governorate": "Metn",
        "population": 25000,
        "social_activity": 68,
        "traffic_index": 50,
        "coffee_shops": 15,
        "sweet_bakery_shops": 10,
        "has_conut": 0,
        "university_presence": 1,
        "tourism_score": 7,
        "avg_rent_index": 68,
    },
    {
        "area": "Hazmieh/Baabda",
        "governorate": "Baabda",
        "population": 80000,
        "social_activity": 65,
        "traffic_index": 78,
        "coffee_shops": 20,
        "sweet_bakery_shops": 15,
        "has_conut": 0,
        "university_presence": 1,
        "tourism_score": 3,
        "avg_rent_index": 60,
    },
    {
        "area": "Aley",
        "governorate": "Aley",
        "population": 90000,
        "social_activity": 55,
        "traffic_index": 60,
        "coffee_shops": 12,
        "sweet_bakery_shops": 10,
        "has_conut": 0,
        "university_presence": 1,
        "tourism_score": 5,
        "avg_rent_index": 45,
    },
    {
        "area": "Chouf (Deir el Qamar)",
        "governorate": "Chouf",
        "population": 70000,
        "social_activity": 40,
        "traffic_index": 35,
        "coffee_shops": 8,
        "sweet_bakery_shops": 10,
        "has_conut": 0,
        "university_presence": 0,
        "tourism_score": 6,
        "avg_rent_index": 35,
    },

    # ── NORTH LEBANON ──
    {
        "area": "Tripoli",
        "governorate": "North",
        "population": 730000,
        "social_activity": 75,
        "traffic_index": 78,
        "coffee_shops": 40,
        "sweet_bakery_shops": 80,  # "Capital of Oriental Sweets"
        "has_conut": 0,
        "university_presence": 2,  # UOB, various
        "tourism_score": 7,
        "avg_rent_index": 35,
    },
    {
        "area": "Batroun",
        "governorate": "North",
        "population": 30000,
        "social_activity": 70,
        "traffic_index": 45,
        "coffee_shops": 18,
        "sweet_bakery_shops": 8,
        "has_conut": 1,  # EXCLUDED — Main Street Coffee
        "university_presence": 0,
        "tourism_score": 8,
        "avg_rent_index": 55,
    },
    {
        "area": "Akkar (Halba)",
        "governorate": "Akkar",
        "population": 320000,
        "social_activity": 30,
        "traffic_index": 40,
        "coffee_shops": 8,
        "sweet_bakery_shops": 15,
        "has_conut": 0,
        "university_presence": 0,
        "tourism_score": 2,
        "avg_rent_index": 20,
    },

    # ── SOUTH LEBANON ──
    {
        "area": "Sidon (Saida)",
        "governorate": "South",
        "population": 170000,
        "social_activity": 60,
        "traffic_index": 65,
        "coffee_shops": 18,
        "sweet_bakery_shops": 25,
        "has_conut": 0,
        "university_presence": 1,
        "tourism_score": 6,
        "avg_rent_index": 40,
    },
    {
        "area": "Tyre (Sour)",
        "governorate": "South",
        "population": 60000,
        "social_activity": 50,
        "traffic_index": 50,
        "coffee_shops": 10,
        "sweet_bakery_shops": 12,
        "has_conut": 1,  # EXCLUDED — Conut - Tyre
        "university_presence": 0,
        "tourism_score": 7,
        "avg_rent_index": 35,
    },
    {
        "area": "Nabatieh",
        "governorate": "Nabatieh",
        "population": 120000,
        "social_activity": 40,
        "traffic_index": 50,
        "coffee_shops": 10,
        "sweet_bakery_shops": 15,
        "has_conut": 0,
        "university_presence": 1,
        "tourism_score": 3,
        "avg_rent_index": 30,
    },

    # ── BEKAA ──
    {
        "area": "Zahle",
        "governorate": "Bekaa",
        "population": 120000,
        "social_activity": 55,
        "traffic_index": 55,
        "coffee_shops": 15,
        "sweet_bakery_shops": 20,
        "has_conut": 0,
        "university_presence": 1,  # Holy Spirit Uni
        "tourism_score": 6,
        "avg_rent_index": 40,
    },
    {
        "area": "Baalbek",
        "governorate": "Baalbek-Hermel",
        "population": 80000,
        "social_activity": 35,
        "traffic_index": 40,
        "coffee_shops": 6,
        "sweet_bakery_shops": 10,
        "has_conut": 0,
        "university_presence": 0,
        "tourism_score": 8,  # Temple ruins, festivals
        "avg_rent_index": 25,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

LOCATION_CONFIG = {
    # Weights for demand proxy calculation
    "demand_weights": {
        "population": 0.35,
        "social_activity": 0.25,
        "traffic_index": 0.20,
        "university_presence": 0.10,
        "tourism_score": 0.10,
    },
    # Ridge regression lambda candidates for MLflow tuning
    "ridge_lambda_candidates": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    # Polynomial degree for feature interactions
    "poly_degree": 2,
    # Scoring weights for final recommendation
    "score_weights": {
        "gap_score": 0.40,       # How underserved the area is
        "demand_score": 0.25,    # How much demand exists
        "affordability": 0.15,   # Inverse of rent (lower rent = better)
        "growth_potential": 0.20, # University + tourism + social
    },
}


def load_area_dataset():
    """
    Load the Lebanese area dataset as a DataFrame.
    Returns DataFrame with all area features.
    """
    df = pd.DataFrame(AREA_DATA)
    df["total_competitors"] = df["coffee_shops"] + df["sweet_bakery_shops"]
    logger.info(f"Loaded {len(df)} Lebanese areas ({df['has_conut'].sum()} with existing Conut branches)")
    return df


def compute_demand_proxy(df):
    """
    Compute a normalized demand proxy for each area based on weighted factors.

    The demand proxy combines population, social activity, traffic, university
    presence, and tourism into a single score representing expected demand.
    """
    weights = LOCATION_CONFIG["demand_weights"]

    # Normalize each factor to 0-1 range
    factors = {}
    for col in ["population", "social_activity", "traffic_index"]:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            factors[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            factors[col] = pd.Series(0.5, index=df.index)

    # University: already 0-3, normalize to 0-1
    factors["university_presence"] = df["university_presence"] / 3.0
    # Tourism: already 0-10, normalize to 0-1
    factors["tourism_score"] = df["tourism_score"] / 10.0

    # Weighted sum
    demand = sum(factors[k] * weights[k] for k in weights)

    # Scale to 0-100
    d_min, d_max = demand.min(), demand.max()
    if d_max > d_min:
        demand = (demand - d_min) / (d_max - d_min) * 100
    else:
        demand = pd.Series(50.0, index=df.index)

    return demand.round(2)


def fit_competitor_model(df):
    """
    Train a Ridge Regression model to predict expected competitor count
    based on area features. Uses all areas (including Conut locations)
    as training data to learn the demand-to-competitor relationship.

    This model captures the NON-LINEAR relationship between demand factors
    and competitor density, which is more accurate than assuming linearity.

    Returns:
        dict with model info, predictions, metrics, and feature importances
    """
    feature_cols = ["population", "social_activity", "traffic_index",
                    "university_presence", "tourism_score"]
    X = df[feature_cols].values
    y = df["total_competitors"].values

    best_model = None
    best_r2 = -999
    best_lambda = None
    best_X_poly = None
    best_scaler = None
    best_poly = None

    if HAS_SKLEARN:
        # Use sklearn with MLflow tuning
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        poly = PolynomialFeatures(degree=LOCATION_CONFIG["poly_degree"],
                                  include_bias=False)
        X_poly = poly.fit_transform(X_scaled)

        if HAS_MLFLOW:
            try:
                mlflow.set_experiment("conut_location_recommender")
            except Exception:
                pass

        for lam in LOCATION_CONFIG["ridge_lambda_candidates"]:
            model = Ridge(alpha=lam, random_state=RANDOM_SEED)
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            if HAS_MLFLOW:
                try:
                    with mlflow.start_run(run_name=f"ridge_lambda_{lam}"):
                        mlflow.log_param("lambda", lam)
                        mlflow.log_param("poly_degree", LOCATION_CONFIG["poly_degree"])
                        mlflow.log_metric("r2", r2)
                        mlflow.log_metric("mae", mae)
                except Exception:
                    pass

            logger.info(f"  Ridge lambda={lam}: R²={r2:.4f}, MAE={mae:.2f}")

            if r2 > best_r2:
                best_r2 = r2
                best_lambda = lam
                best_model = model
                best_X_poly = X_poly
                best_scaler = scaler
                best_poly = poly

        y_pred = best_model.predict(best_X_poly)

        # Feature importances (from polynomial features)
        feat_names = best_poly.get_feature_names_out(feature_cols)
        importances = np.abs(best_model.coef_)
        imp_sum = importances.sum()
        if imp_sum > 0:
            importances = importances / imp_sum

        feature_importance = sorted(
            zip(feat_names, importances),
            key=lambda x: x[1], reverse=True
        )[:10]

    else:
        # Numpy fallback: simple polynomial regression
        # Normalize features manually
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1
        X_scaled = (X - X_mean) / X_std

        # Add polynomial interactions (degree 2)
        n = X_scaled.shape[0]
        poly_features = [X_scaled]
        for i in range(X_scaled.shape[1]):
            for j in range(i, X_scaled.shape[1]):
                poly_features.append((X_scaled[:, i] * X_scaled[:, j]).reshape(-1, 1))
        X_poly = np.hstack(poly_features)

        # Add bias
        X_bias = np.hstack([np.ones((n, 1)), X_poly])

        best_lambda = 1.0
        for lam in LOCATION_CONFIG["ridge_lambda_candidates"]:
            I = np.eye(X_bias.shape[1])
            I[0, 0] = 0  # Don't regularize bias
            w = np.linalg.solve(X_bias.T @ X_bias + lam * I, X_bias.T @ y)
            y_pred = X_bias @ w
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            mae = np.mean(np.abs(y - y_pred))

            logger.info(f"  Ridge lambda={lam}: R²={r2:.4f}, MAE={mae:.2f}")

            if r2 > best_r2:
                best_r2 = r2
                best_lambda = lam
                best_model = w
                best_X_poly = X_bias

        y_pred = best_X_poly @ best_model
        feature_importance = list(zip(feature_cols, [0.2] * len(feature_cols)))
        best_scaler = {"mean": X_mean, "std": X_std}
        best_poly = None

    logger.info(f"\n  Best model: lambda={best_lambda}, R²={best_r2:.4f}")

    return {
        "model": best_model,
        "scaler": best_scaler,
        "poly": best_poly,
        "best_lambda": best_lambda,
        "r2": round(best_r2, 4),
        "mae": round(np.mean(np.abs(y - y_pred)), 2),
        "y_pred": y_pred,
        "y_actual": y,
        "feature_importance": feature_importance,
        "feature_cols": feature_cols,
    }


def compute_gap_scores(df, model_result):
    """
    Compute the market gap for each area:
    gap = expected_competitors (from model) - actual_competitors

    Positive gap = area is UNDERSERVED (opportunity!)
    Negative gap = area is OVERSATURATED
    """
    expected = model_result["y_pred"]
    actual = df["total_competitors"].values

    gap = expected - actual
    # Normalize gap to 0-100 scale
    gap_min, gap_max = gap.min(), gap.max()
    if gap_max > gap_min:
        gap_normalized = (gap - gap_min) / (gap_max - gap_min) * 100
    else:
        gap_normalized = np.full_like(gap, 50.0)

    return gap, gap_normalized


def compute_final_scores(df, demand_proxy, gap_normalized):
    """
    Compute final recommendation scores combining:
    - Gap score (how underserved the area is)
    - Demand score (raw demand proxy)
    - Affordability (inverse of rent index)
    - Growth potential (university + tourism + social activity)
    """
    weights = LOCATION_CONFIG["score_weights"]

    # Normalize demand to 0-100 (already done)
    demand_score = demand_proxy

    # Affordability: invert rent index
    rent = df["avg_rent_index"].values.astype(float)
    affordability = 100 - rent  # Lower rent = higher score

    # Growth potential
    growth = (
        df["university_presence"].values / 3.0 * 30 +
        df["tourism_score"].values / 10.0 * 40 +
        df["social_activity"].values / 100.0 * 30
    )

    final = (
        weights["gap_score"] * gap_normalized +
        weights["demand_score"] * demand_score +
        weights["affordability"] * affordability +
        weights["growth_potential"] * growth
    )

    return final.round(2)


def run_full_location_analysis():
    """
    Complete location recommendation pipeline.

    Returns:
        dict with ranked areas, model metrics, and recommendations
    """
    logger.info("=" * 60)
    logger.info("BRANCH LOCATION RECOMMENDER")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("\n[1/5] Loading Lebanese area dataset...")
    df = load_area_dataset()

    # Step 2: Compute demand proxy
    logger.info("\n[2/5] Computing demand proxy for each area...")
    df["demand_proxy"] = compute_demand_proxy(df)

    # Step 3: Train competitor prediction model
    logger.info("\n[3/5] Training competitor prediction model (Ridge Regression)...")
    model_result = fit_competitor_model(df)
    df["expected_competitors"] = model_result["y_pred"].round(1)

    # Step 4: Compute gap scores
    logger.info("\n[4/5] Computing market gap scores...")
    gap_raw, gap_normalized = compute_gap_scores(df, model_result)
    df["competitor_gap"] = gap_raw.round(1)
    df["gap_score"] = gap_normalized.round(1)

    # Step 5: Compute final recommendation scores
    logger.info("\n[5/5] Computing final recommendation scores...")
    df["final_score"] = compute_final_scores(df, df["demand_proxy"].values, gap_normalized)

    # Filter out existing Conut locations
    available = df[df["has_conut"] == 0].copy()
    available = available.sort_values("final_score", ascending=False).reset_index(drop=True)
    available.index = available.index + 1
    available.index.name = "rank"

    # Build summary
    summary = _build_summary(available, model_result, df)

    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)

    return {
        "rankings": available,
        "all_areas": df,
        "model_metrics": {
            "r2": model_result["r2"],
            "mae": model_result["mae"],
            "best_lambda": model_result["best_lambda"],
            "feature_importance": model_result["feature_importance"],
        },
        "summary": summary,
    }


def _build_summary(ranked, model_result, all_df):
    """Build human-readable summary of the analysis."""
    lines = [
        "BRANCH LOCATION RECOMMENDATION SUMMARY",
        "=" * 50,
        "",
        f"Areas analyzed: {len(all_df)}",
        f"Existing Conut branches (excluded): {int(all_df['has_conut'].sum())}",
        f"Candidate areas: {len(ranked)}",
        "",
        f"Competitor Model R²: {model_result['r2']:.4f}",
        f"Competitor Model MAE: {model_result['mae']:.1f} shops",
        f"Best Ridge Lambda: {model_result['best_lambda']}",
        "",
        "TOP 5 RECOMMENDED LOCATIONS:",
        "-" * 50,
    ]

    for i, (_, row) in enumerate(ranked.head(5).iterrows()):
        gap_dir = "UNDERSERVED" if row["competitor_gap"] > 0 else "saturated"
        lines.append(
            f"\n  #{i+1}. {row['area']} ({row['governorate']})"
        )
        lines.append(f"      Final Score: {row['final_score']:.1f}/100")
        lines.append(f"      Population: {row['population']:,}")
        lines.append(f"      Competitors: {row['total_competitors']} actual vs "
                     f"{row['expected_competitors']:.0f} expected → {gap_dir} by "
                     f"{abs(row['competitor_gap']):.0f} shops")
        lines.append(f"      Demand Proxy: {row['demand_proxy']:.1f}/100")
        lines.append(f"      Rent Index: {row['avg_rent_index']}/100")

    # Also show areas to AVOID
    bottom = ranked.tail(3)
    lines.append("\n\nAREAS TO AVOID (oversaturated or low demand):")
    lines.append("-" * 50)
    for _, row in bottom.iterrows():
        lines.append(f"  - {row['area']}: score {row['final_score']:.1f}/100, "
                     f"gap={row['competitor_gap']:.0f}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# OPENCLAW AGENT TOOL
# ══════════════════════════════════════════════════════════════════════════════

def get_branch_location_recommendation(top_n=5, governorate=None, min_population=None):
    """
    OpenClaw-compatible function: Recommends the best Lebanese areas
    to open a new Conut branch based on market gap analysis.

    Parameters
    ----------
    top_n : int
        Number of top recommendations to return (default 5)
    governorate : str or None
        Filter to a specific governorate (e.g., "Beirut", "North", "Bekaa")
    min_population : int or None
        Minimum population threshold for candidate areas

    Returns
    -------
    dict with status, recommendations, model metrics, and summary
    """
    try:
        result = run_full_location_analysis()
        ranked = result["rankings"]

        if governorate:
            ranked = ranked[ranked["governorate"].str.lower() == governorate.lower()]
        if min_population:
            ranked = ranked[ranked["population"] >= min_population]

        recommendations = []
        for _, row in ranked.head(top_n).iterrows():
            gap_direction = "underserved" if row["competitor_gap"] > 0 else "oversaturated"
            recommendations.append({
                "area": row["area"],
                "governorate": row["governorate"],
                "final_score": round(row["final_score"], 1),
                "population": int(row["population"]),
                "total_competitors": int(row["total_competitors"]),
                "expected_competitors": round(row["expected_competitors"], 0),
                "competitor_gap": round(row["competitor_gap"], 1),
                "gap_direction": gap_direction,
                "demand_proxy": round(row["demand_proxy"], 1),
                "rent_index": int(row["avg_rent_index"]),
                "social_activity": int(row["social_activity"]),
                "university_presence": int(row["university_presence"]),
                "tourism_score": int(row["tourism_score"]),
            })

        return {
            "status": "success",
            "recommendations": recommendations,
            "total_areas_analyzed": len(result["all_areas"]),
            "existing_branches_excluded": ["Tyre (Conut - Tyre)",
                                            "Jnah (Conut Jnah)",
                                            "Batroun (Main Street Coffee)"],
            "model_r2": result["model_metrics"]["r2"],
            "model_mae": result["model_metrics"]["mae"],
            "summary": result["summary"],
        }

    except Exception as e:
        logger.error(f"Location recommender error: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import json
    result = get_branch_location_recommendation(top_n=5)
    print(json.dumps(result, indent=2, default=str))
