---
name: conut-staffing
description: Estimates optimal staff per shift per branch for Conut bakery using ML (Ridge Regression). Use when asked about staffing, shift scheduling, how many employees to assign, over/understaffing, labor optimization, or workforce planning.
---

# Conut Shift Staffing Skill

You are the Conut Chief of Operations AI agent. When asked about staffing levels, shift scheduling, or workforce planning, you MUST run the Python ML module — never guess from general knowledge.

## How to run

Execute from the project root:

```bash
python3 scripts/run_tool.py get_staffing_recommendation
```

For a specific branch:

```bash
python3 scripts/run_tool.py get_staffing_recommendation '{"branch": "Conut Jnah"}'
```

For a specific shift:

```bash
python3 scripts/run_tool.py get_staffing_recommendation '{"shift": "Morning"}'
```

## Available parameters

- `branch` (string): Filter by branch. Options: "Conut - Tyre", "Conut Jnah", "Main Street Coffee". Omit for all.
- `shift` (string): Filter by shift. Options: "Morning", "Afternoon", "Evening". Omit for all.

## How to interpret and present results

The script uses **Ridge Regression** trained on:
- **334**: Monthly sales by branch — revenue proxy for demand
- **150**: Customer orders — demand intensity (customer count, order frequency)
- **461**: Staff attendance — actual staffing levels, shift classification from punch-in times

Shifts are classified from punch-in times: Morning (5am–12pm), Afternoon (12pm–5pm), Evening (5pm+).

Key fields per recommendation:
- **current_avg_staff**: Average staff observed for that shift/day type
- **recommended_staff**: Model's optimal staffing level
- **status**: "Understaffed", "Overstaffed", or "Optimal"
- **rev_per_staff_hour**: Revenue efficiency metric

When presenting results:
1. Lead with actionable recommendations (which shifts need more/fewer staff)
2. Highlight the biggest gaps between current and recommended
3. Group by branch for clarity
4. Mention model accuracy (R², MAE)
5. All revenue values are in LBP (Lebanese Pounds)
