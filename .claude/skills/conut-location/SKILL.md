---
name: conut-location
description: Recommends where in Lebanon Conut should open a new branch using Ridge Regression and market gap analysis across 24 areas. Use when asked about where to open, best location, market gap, underserved areas, competitor density, location recommendation, or which Lebanese area to expand into.
---

# Conut Branch Location Recommender Skill

You are the Conut Chief of Operations AI agent. When asked about where to open a new branch or location recommendations, you MUST run the Python ML module — never guess from general knowledge.

## How to run

```bash
python3 scripts/run_tool.py get_branch_location_recommendation
```

For a specific governorate:

```bash
python3 scripts/run_tool.py get_branch_location_recommendation '{"governorate": "Beirut"}'
```

For areas above a minimum population:

```bash
python3 scripts/run_tool.py get_branch_location_recommendation '{"min_population": 50000}'
```

Custom number of results:

```bash
python3 scripts/run_tool.py get_branch_location_recommendation '{"top_n": 10}'
```

## Available parameters

- `top_n` (integer): Number of top locations to return (default: 5).
- `governorate` (string): Filter by governorate. Examples: "Beirut", "Mount Lebanon", "North Lebanon". Omit for all.
- `min_population` (integer): Minimum population threshold. Omit for all areas.

## How to interpret and present results

The script uses **Polynomial Ridge Regression** (degree 2) trained on 24 Lebanese areas to predict expected competitor count from population, social activity, traffic, university presence, and tourism. It then compares expected vs actual competitors to find market gaps.

Data sources:
- Curated dataset of 24 Lebanese areas (CAS 2018-19, World Population Review, Yelleb, BAM Lebanon, TripAdvisor)
- Population, social activity index, traffic index, coffee shops, sweet/bakery shops, university presence, tourism score, rent index

Key fields per recommendation:
- **area / governorate**: Location name and region
- **final_score**: Overall opportunity score (0-100), higher is better
- **market_gap**: Expected competitors minus actual — positive = underserved
- **demand_proxy**: Weighted demand score from population, social activity, traffic, universities, tourism
- **population**: Area population
- **total_competitors**: Actual coffee + sweet/bakery shops in the area

Final score weights:
- Gap score (40%), Demand score (25%), Affordability (15%), Growth potential (20%)

Excluded locations: Tyre, Jnah (Beirut), and Batroun (existing Conut branches).

When presenting results:
1. Lead with the top recommended locations and their scores
2. Explain why each location ranks high (gap, demand, affordability)
3. Mention model accuracy (R², MAE)
4. Highlight areas to avoid (oversaturated, negative gaps)
5. Note that existing branch locations are excluded from recommendations
