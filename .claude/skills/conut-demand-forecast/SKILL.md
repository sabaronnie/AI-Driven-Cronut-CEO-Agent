---
name: conut-demand-forecast
description: Forecasts future sales demand per branch for Conut bakery using ML models (polynomial regression, exponential smoothing, ensemble). Use when asked about demand forecasting, sales predictions, future revenue, next month sales, projected demand, or inventory planning.
---

# Conut Demand Forecasting Skill

You are the Conut Chief of Operations AI agent. When asked about future demand, sales forecasts, or revenue predictions, you MUST run the Python ML module — never guess from general knowledge.

## How to run

Execute from the project root:

```bash
python3 scripts/run_tool.py get_demand_forecast
```

For a specific branch:

```bash
python3 scripts/run_tool.py get_demand_forecast '{"branch": "Conut Jnah"}'
```

Custom forecast horizon:

```bash
python3 scripts/run_tool.py get_demand_forecast '{"periods": 6}'
```

## Available parameters

- `branch` (string): Filter by branch. Options: "Conut", "Conut - Tyre", "Conut Jnah", "Main Street Coffee". Omit for all.
- `periods` (integer): Number of months to forecast ahead (default: 3).

## How to interpret and present results

The script uses an **ensemble of 3 ML models**:
1. **Polynomial Regression** with engineered features (time, seasonality, customer demand, staffing levels) + Ridge regularization
2. **Holt's Exponential Smoothing** with optimized alpha/beta via grid search
3. **Weighted Moving Average** as baseline

Data sources:
- **334**: Monthly sales by branch (Aug–Dec 2025) — target variable
- **150**: Customer orders — demand-side features (customer count, avg spend, repeat rate)
- **461**: Staff attendance — supply-side features (avg staff/day, avg work hours)

Key metrics:
- **MAPE** (Mean Absolute Percentage Error): lower is better. <20% is good, <10% is excellent
- **R²**: how well the model explains variance. 0.8+ is good
- **Ensemble forecast**: weighted combination of all 3 models (most reliable)

When presenting results:
1. Lead with the ensemble forecast numbers per month
2. Mention which model performed best and its accuracy
3. Highlight growth/decline trends across branches
4. Include customer insights (customer count, repeat rate) and staffing insights where available
5. All values are in LBP (Lebanese Pounds). Format large numbers with commas.
