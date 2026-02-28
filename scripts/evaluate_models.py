#!/usr/bin/env python3
"""
Standalone script to evaluate and display accuracy metrics for all ML models.
Demonstrates model performance for hackathon judges.

Usage:
    cd /path/to/Hackaton
    python3 scripts/evaluate_models.py
"""

import os
import sys
import json
import logging

# Setup path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))
os.chdir(project_root)

# Suppress logger output from modules (we want clean metrics only)
logging.disable(logging.WARNING)

from demand_forecasting import run_full_demand_forecast, forecast_branch, load_data, prepare_time_series
from shift_staffing import run_full_staffing_analysis
from expansion_feasibility import run_full_expansion_analysis
from branch_location_recommender import run_full_location_analysis


def header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def subheader(text):
    print(f"\n  {text}")
    print("  " + "-" * 70)


def metric(name, value, width=35):
    if isinstance(value, float):
        if abs(value) > 1000:
            print(f"    {name:<{width}}: {value:>15,.0f}")
        else:
            print(f"    {name:<{width}}: {value:>15.4f}")
    else:
        print(f"    {name:<{width}}: {str(value):>15}")


def evaluate_demand_forecasting():
    """Evaluate Model 2: Demand Forecasting."""
    header("MODEL 2: DEMAND FORECASTING (Polynomial Ridge Regression + Holt's ES + WMA Ensemble)")

    logging.disable(logging.CRITICAL)
    result = run_full_demand_forecast()
    logging.disable(logging.WARNING)

    if "branch_forecasts" not in result:
        print("    Error running demand forecast")
        return

    for branch_name, forecast in result["branch_forecasts"].items():
        subheader(f"Branch: {branch_name}")

        if forecast["status"] != "success":
            print(f"    {forecast.get('message', 'Error')}")
            continue

        # Polynomial Regression metrics
        poly = forecast["models"]["polynomial_regression"]["metrics"]
        print("    Polynomial Regression (Ridge):")
        metric("R²", poly["r_squared"])
        metric("MAPE (%)", poly["mape"])
        metric("MAE (LBP)", poly["mae"])
        metric("RMSE (LBP)", poly["rmse"])

        # Holt's ES metrics
        holt = forecast["models"]["exponential_smoothing"]["metrics"]
        holt_params = forecast["models"]["exponential_smoothing"].get("optimal_params", {})
        print("\n    Holt's Exponential Smoothing:")
        metric("MAPE (%)", holt["mape"])
        metric("MAE (LBP)", holt["mae"])
        if holt_params:
            metric("Optimal alpha", holt_params.get("alpha", "N/A"))
            metric("Optimal beta", holt_params.get("beta", "N/A"))

        # Best model
        poly_mape = poly["mape"]
        holt_mape = holt["mape"]
        best = "Polynomial Regression" if poly_mape < holt_mape else "Exponential Smoothing"
        best_mape = min(poly_mape, holt_mape)
        print(f"\n    >>> Best Model: {best} (MAPE: {best_mape:.2f}%)")

        # LOO-CV metrics if available
        if "cv_mape" in forecast.get("models", {}).get("polynomial_regression", {}):
            cv_mape = forecast["models"]["polynomial_regression"]["cv_mape"]
            print(f"\n    LOO Cross-Validation:")
            metric("CV MAPE (%)", cv_mape)

        # Data summary
        print(f"\n    Data: {forecast['data_points']} months, "
              f"Customers(150): {forecast['metadata']['n_customers']}, "
              f"Staff/day(461): {forecast['metadata']['avg_staff_per_day']:.1f}")


def evaluate_expansion_feasibility():
    """Evaluate Model 3: Expansion Feasibility."""
    header("MODEL 3: EXPANSION FEASIBILITY (K-Means Clustering + Weighted Scoring)")

    logging.disable(logging.CRITICAL)
    result = run_full_expansion_analysis()
    logging.disable(logging.WARNING)

    if result.get("status") != "success":
        print(f"    Error: {result.get('message', 'Unknown')}")
        return

    # Clustering results
    subheader("K-Means Clustering Results")
    clustering = result["clustering"]
    metric("Features used", ", ".join(clustering["features_used"]))
    metric("Number of clusters", clustering["n_clusters"])

    if "silhouette_score" in clustering:
        metric("Silhouette Score", clustering["silhouette_score"])

    print("\n    Cluster Assignments:")
    for branch, label in clustering["assignments"].items():
        print(f"      {branch:<30}: {label}")

    # Expansion score breakdown
    subheader("Expansion Feasibility Score")
    score = result["expansion_score"]
    metric("TOTAL SCORE", f"{score['total_score']}/100")
    metric("Recommendation", score["recommendation"])

    print("\n    Dimension Breakdown:")
    for dim, val in score["dimensions"].items():
        weight = score["weights"].get(dim, 0)
        bar = "█" * int(val / 5) + "░" * (20 - int(val / 5))
        print(f"      {dim:<30}: {val:>5.1f}/100  [{bar}]  (weight: {weight:.0%})")


def evaluate_shift_staffing():
    """Evaluate Model 4: Shift Staffing."""
    header("MODEL 4: SHIFT STAFFING (Ridge Regression)")

    logging.disable(logging.CRITICAL)
    result = run_full_staffing_analysis()
    logging.disable(logging.WARNING)

    if result.get("status") != "success":
        print(f"    Error: {result.get('message', 'Unknown')}")
        return

    subheader("Model Performance Metrics")
    metrics = result["model_metrics"]
    metric("R²", metrics["r_squared"])
    metric("MAE (staff)", metrics["mae"])
    metric("RMSE (staff)", metrics["rmse"])
    metric("Training samples", metrics["n_samples"])

    # Feature importances
    subheader("Feature Importances")
    importances = metrics["feature_importances"]
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_feats:
        bar = "█" * int(imp * 40) + "░" * (40 - int(imp * 40))
        print(f"    {feat:<30}: {imp:.3f}  [{bar}]")

    # CV results if available
    if "cv_results" in metrics:
        cv = metrics["cv_results"]
        subheader("Cross-Validation Results (K-Fold)")
        if "mean_r2" in cv:
            metric("CV R² (mean ± std)", f"{cv['mean_r2']:.3f} ± {cv.get('std_r2', 0):.3f}")
        if "mean_mae" in cv:
            metric("CV MAE (mean ± std)", f"{cv['mean_mae']:.3f} ± {cv.get('std_mae', 0):.3f}")

    # Staffing summary
    subheader("Staffing Recommendations Summary")
    recs = result["recommendations"]
    understaffed = sum(1 for r in recs if r["status"] == "Understaffed")
    overstaffed = sum(1 for r in recs if r["status"] == "Overstaffed")
    optimal = sum(1 for r in recs if r["status"] == "Optimal")
    total = len(recs)
    print(f"    Understaffed: {understaffed}/{total}  |  Overstaffed: {overstaffed}/{total}  |  Optimal: {optimal}/{total}")


def evaluate_location_recommender():
    """Evaluate Model 6: Branch Location Recommender."""
    header("MODEL 6: BRANCH LOCATION RECOMMENDER (Ridge Regression + Gap Analysis)")

    logging.disable(logging.CRITICAL)
    result = run_full_location_analysis()
    logging.disable(logging.WARNING)

    rankings = result["rankings"]
    metrics = result["model_metrics"]

    subheader("Competitor Prediction Model")
    metric("R²", metrics["r2"])
    metric("MAE (shops)", metrics["mae"])
    metric("Best Ridge Lambda", metrics["best_lambda"])
    metric("Areas analyzed", len(result["all_areas"]))
    metric("Candidate areas", len(rankings))

    subheader("Top Feature Importances")
    for feat, imp in metrics["feature_importance"][:8]:
        bar = "█" * int(imp * 40) + "░" * (40 - int(imp * 40))
        print(f"    {feat:<30}: {imp:.3f}  [{bar}]")

    subheader("Top 5 Recommended Locations")
    for i, (_, row) in enumerate(rankings.head(5).iterrows()):
        gap_dir = "UNDERSERVED" if row["competitor_gap"] > 0 else "saturated"
        gap_abs = abs(row["competitor_gap"])
        print(f"\n    #{i+1}. {row['area']} ({row['governorate']})")
        print(f"        Score: {row['final_score']:.1f}/100  |  Pop: {row['population']:,}  |  "
              f"Competitors: {row['total_competitors']} actual vs {row['expected_competitors']:.0f} expected")
        print(f"        Gap: {gap_dir} by {gap_abs:.0f} shops  |  "
              f"Demand: {row['demand_proxy']:.1f}/100  |  Rent: {row['avg_rent_index']}/100")

    subheader("Excluded (Existing Conut Branches)")
    excluded = result["all_areas"][result["all_areas"]["has_conut"] == 1]
    for _, row in excluded.iterrows():
        print(f"    - {row['area']} ({row['governorate']})")


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "CONUT AI CHIEF OF OPERATIONS — MODEL EVALUATION" + " " * 15 + "║")
    print("║" + " " * 10 + "Comprehensive accuracy metrics for hackathon judges" + " " * 17 + "║")
    print("╚" + "═" * 78 + "╝")

    evaluate_demand_forecasting()
    evaluate_expansion_feasibility()
    evaluate_shift_staffing()
    evaluate_location_recommender()

    header("EVALUATION COMPLETE")
    print("  Models 2, 3, 4, and 6 use ML (Ridge Regression, Holt's ES, K-Means).")
    print("  Models 1 and 5 use analytics (Apriori association rules, BI aggregation).")
    print("\n  Run individual tools:  python3 scripts/run_tool.py <tool_name>")
    print("  Run unit tests:        python3 -m unittest tests.test_models -v")
    print()


if __name__ == "__main__":
    main()
