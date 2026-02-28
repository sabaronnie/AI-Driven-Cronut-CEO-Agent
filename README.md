# Conut AI-Driven Chief of Operations

## Overview

A 12-hour hackathon project building an intelligent AI operations agent for Conut bakery chain in Lebanon. The system uses machine learning and data analytics to optimize business operations across six dimensions: product combo optimization, demand forecasting, expansion feasibility, shift staffing, beverage growth strategy, and branch location scouting.

Features a web-based chat interface powered by Claude (Anthropic) with 6 ML tool integrations. Also compatible with **OpenClaw** via skill definitions.

## Business Objectives

| # | Model | Method | Question It Answers |
|---|-------|--------|---------------------|
| 1 | Combo Optimization | Apriori association rules | "What product bundles should we promote?" |
| 2 | Demand Forecasting | Polynomial Ridge Regression + Holt's ES + WMA Ensemble | "What will our sales look like next quarter?" |
| 3 | Expansion Feasibility | K-Means Clustering + Weighted Scoring | "Should Conut open a new branch?" |
| 4 | Shift Staffing | Ridge Regression with polynomial features | "How many staff do we need per shift?" |
| 5 | Coffee & Milkshake Growth | BI analytics + cross-sell analysis | "How do we grow coffee and milkshake revenue?" |
| 6 | Branch Location Recommender | Ridge Regression + Market Gap Analysis | "Where in Lebanon should we open next?" |

## Architecture

```
                    ┌─────────────────────────────┐
                    │     Frontend (Chat UI)       │
                    │   HTML / CSS / JavaScript    │
                    └──────────┬──────────────────-┘
                               │  POST /api/chat
                               ▼
                    ┌─────────────────────────────┐
                    │   Backend (FastAPI Server)   │
                    │   Anthropic Claude API +     │
                    │   Tool-use loop              │
                    └──────────┬──────────────────-┘
                               │  subprocess
                               ▼
                    ┌─────────────────────────────┐
                    │   scripts/run_tool.py        │
                    │   Routes to 6 ML modules     │
                    └──────────┬──────────────────-┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  src/*.py     │  │  data/       │  │  .claude/    │
    │  6 ML models  │  │  8 CSVs      │  │  skills/     │
    └──────────────┘  └──────────────┘  └──────────────┘
```

## Tech Stack

- **Python 3.10+**
- **FastAPI** + **Uvicorn**: Backend API server
- **Anthropic SDK**: Claude API with tool-use for AI reasoning
- **pandas** / **numpy**: Data manipulation and numerical computation
- **scikit-learn**: KMeans clustering, silhouette score, regression metrics
- **MLflow**: Hyperparameter tuning and experiment tracking
- **OpenClaw** (optional): Agent deployment with skill definitions

## Project Structure

```
Hackaton/
├── frontend/                         # Chat UI
│   ├── index.html                    # Main page
│   ├── css/styles.css                # Styling
│   ├── js/app.js                     # Chat logic + API calls
│   └── assets/                       # Static assets
├── backend/                          # API server
│   ├── server.py                     # FastAPI app with Claude tool-use
│   ├── requirements.txt              # All Python dependencies
│   ├── .env.example                  # API key template
│   └── .env                          # Your API key (create this)
├── src/                              # ML models (do not modify)
│   ├── config.py                     # Centralized config + validation
│   ├── combo_optimization.py         # Model 1: Apriori rules
│   ├── demand_forecasting.py         # Model 2: Poly Ridge + Holt's ES
│   ├── expansion_feasibility.py      # Model 3: K-Means + scoring
│   ├── shift_staffing.py             # Model 4: Ridge Regression
│   ├── coffee_milkshake_growth.py    # Model 5: BI analytics
│   └── branch_location_recommender.py# Model 6: Location gap analysis
├── scripts/                          # CLI tools
│   ├── run_tool.py                   # Run any model from command line
│   └── evaluate_models.py            # Display metrics for judges
├── tests/
│   └── test_models.py                # 61 unit tests
├── data/
│   └── cleaned/                      # 8 cleaned CSV data sources
├── Dockerfile                        # Docker container setup
├── docker-compose.yml                # One-command startup
├── .dockerignore                     # Docker build exclusions
├── .claude/skills/                   # OpenClaw skill definitions
├── CLAUDE.md                         # Agent persona (OpenClaw)
├── start.sh                          # One-command startup
└── README.md
```

## Quick Start — Docker (Recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### First time (build + run)

**macOS / Linux:**
```bash
cd Hackaton
export ANTHROPIC_API_KEY=sk-ant-xxxxx
docker compose up --build
```

**Windows (PowerShell):**
```powershell
cd Hackaton
$env:ANTHROPIC_API_KEY="sk-ant-xxxxx"
docker compose up --build
```

Open **http://localhost:8000** in your browser. That's it.

### Reopening after first build (no rebuild needed)

**macOS / Linux:**
```bash
cd Hackaton
export ANTHROPIC_API_KEY=sk-ant-xxxxx
docker compose up
```

**Windows (PowerShell):**
```powershell
cd Hackaton
$env:ANTHROPIC_API_KEY="sk-ant-xxxxx"
docker compose up
```

### Stopping the server

Press `Ctrl+C` in the terminal, or run:
```bash
docker compose down
```

## Alternative — Manual Setup (without Docker)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Install Claude Code (OpenClaw)
npm install -g @anthropic-ai/claude-code

# 4. Configure
cp backend/.env.example backend/.env
# Edit backend/.env — add your ANTHROPIC_API_KEY

# 5. Start the server
python3 backend/server.py
# → Open http://localhost:8000
```

## CLI Usage (without frontend)

```bash
# Run individual tools directly
python3 scripts/run_tool.py get_combo_recommendations
python3 scripts/run_tool.py get_demand_forecast
python3 scripts/run_tool.py get_expansion_assessment
python3 scripts/run_tool.py get_staffing_recommendation
python3 scripts/run_tool.py get_coffee_milkshake_analysis
python3 scripts/run_tool.py get_branch_location_recommendation

# Evaluate model accuracy
python3 scripts/evaluate_models.py

# Run unit tests (61 tests)
python3 -m unittest tests.test_models -v
```

## Data Cleaning & Preprocessing

All raw data was exported from Omega POS (Point of Sale) software as CSV reports. These exports are messy multi-page documents with repeated page headers/footers, copyright lines, inconsistent column layouts across pages, and embedded aggregate rows. Each file required a dedicated Python cleaning script (see `data/cleaning_scripts/`) to produce model-ready datasets.

### Common Issues Across All Files

- **Repeated page headers**: Every ~30 rows, Omega re-inserts date rows, column headers, and page numbers (e.g., `Page 3 of 45`)
- **Copyright footers**: Lines like `REP_S_00136, "Copyright © 2026 Omega Software, Inc."`
- **Inconsistent column layouts**: Some reports change column count mid-file (e.g., page 1 has 11 columns, pages 2+ have 6)
- **Quoted numbers with commas**: Revenue values like `"2,116,800.0"` require comma stripping before numeric conversion
- **Forward-fill needed**: Division/branch names only appear on the first row of each group; subsequent rows are blank
- **Aggregate rows mixed with data**: `TOTAL`, `Total By Branch`, and `Total by Group` rows interspersed with actual data

### Per-File Cleaning: Column Decisions & Feature Engineering

#### 1. REP_S_00136_SMRY → `cleaned_summary.csv`
Revenue by category across delivery/takeaway/table channels per branch. The raw export uses an 11-column wide layout on page 1 that collapses to 6 columns on subsequent pages — we normalized both into a single 6-column schema. Dropped all `TOTAL` aggregate rows (branch-level sums that would double-count in our models) and removed rows where all revenue channels were 0.00 (categories with no sales activity, which add noise to combo and growth analysis). Division names only appear on the first row of each group in the raw file — we forward-filled these to make every row self-contained for groupby operations.

#### 2. rep_s_00150 → `cleaned_customer_orders.csv`
Per-customer delivery order history. Dropped the `Total By Branch` summary rows to avoid inflating aggregates. Removed 24 void/cancelled orders (total = 0.00) since they don't represent real demand. Engineered `first_order` and `last_order` timestamps by cleaning trailing colons from Omega's date format (`2025-12-31 19:04:` → `2025-12-31 19:04`), giving us customer lifetime windows for demand forecasting.

#### 3. rep_s_00191_SMRY → `cleaned_sales_by_item.csv`
Hierarchical Division → Group → Item revenue breakdown. The raw file nests branch, division, and group as stateful headers rather than columns — we flattened this hierarchy into explicit columns via stateful parsing. Dropped all `Total by Group/Division/Branch` aggregate rows to keep only leaf-level item data. Removed 456 zero-revenue rows (free toppings, customization options) that have no value for revenue-based models like combo optimization and coffee/milkshake growth analysis.

#### 4. REP_S_00502 → `cleaned_sales_by_customer.csv`
Line-item detail per customer delivery order. Dropped 1,189 rows with price = 0.00 — these are free add-ons and options that don't contribute to revenue modeling. Kept negative-quantity rows (qty = -1) intentionally, as these represent cancellations and are useful for identifying refund patterns. Engineered per-person totals by buffering items and flushing on the `Total :` delimiter row.

#### 5. rep_s_00435_SMRY → `cleaned_avg_sales_by_menu.csv`
Customer count, revenue, and average spend per channel per branch. Key engineering: used MultiIndex reindexing to ensure every branch has all 3 channels (delivery, takeaway, table), zero-filling missing combinations. This matters because some branches genuinely don't operate certain channels — rather than treating these as missing data, we encode them as 0 to preserve the structural truth for the expansion feasibility model.

#### 6. rep_s_00334_1_SMRY → `cleaned_monthly_sales.csv`
Monthly revenue per branch. Main Street Coffee (Batroun) is missing August 2025 because it opened in September — we zero-filled this rather than interpolating, so the model sees the real operational timeline. Engineered separate `month` and `year` columns from the raw date headers for time-series feature extraction in demand forecasting.

#### 7. REP_S_00461 → `cleaned_attendance.csv`
Employee punch-in/out records. **Dropped employee ID and name columns entirely** — these are PII (personally identifiable information) and irrelevant to our staffing models, which only need shift timing and branch. Converted Omega's `HH.MM.SS` time format to standard `HH:MM:SS` and parsed dates from `01-Dec-25` format to proper datetime objects, enabling `work_duration` computation for shift-level staffing predictions.

#### 8. REP_S_00194_SMRY → `cleaned_tax_report.csv`
Tax report with multiple tax rate columns. **Dropped all tax columns except VAT 11%** — every other tax column (VAT 0%, exempt, etc.) contained only 0.00 values across all branches. Retaining only the active tax column keeps the dataset lean for the expansion feasibility model's financial scoring.

### Data Integrity Notes

- **ITEMS is not an aggregate**: Verified by summing all other categories per division — ITEMS represents core product revenue (chimneys, conuts, minis), while other categories are add-ons/extras/drinks.
- **Missing channels are structural**: Cross-referenced across reports 136, 150, 435, and 502 — the same channels are consistently absent for the same branches, confirming operational structure rather than data gaps.
- **Zero-fill strategy**: Missing months and channels filled with 0 (not interpolated or averaged), preserving data integrity for model training.

## Data Sources

| File | Rows | Used By | Description |
|------|------|---------|-------------|
| `sales_detail_(502).csv` | ~4,700 | Models 1, 5 | Transaction-level data |
| `cleaned_sales_by_item_(191).csv` | ~900 | Models 1, 5 | Product revenue by branch |
| `cleaned_monthly_sales_(334).csv` | ~60 | Models 2, 3 | Monthly revenue per branch |
| `cleaned_avg_sales_by_menu_(435).csv` | ~150 | Model 3 | Average sales by category |
| `customer_orders_(150).csv` | ~150 | Models 2, 3, 4 | Customer orders per month |
| `cleaned_attendance_(461).csv` | ~460 | Models 2, 4 | Staff attendance per branch |
| `cleaned_revenue_summary_(136).csv` | ~130 | Models 3, 5 | Channel breakdown |
| `cleaned_tax_report_(194).csv` | ~190 | Model 3 | Tax data per branch |

## Key Metrics

- **Model 2**: Per-branch MAPE typically 5-15%, LOO-CV validates generalization
- **Model 3**: Silhouette score ~0.65, expansion score 67/100
- **Model 4**: R² ~0.28, MAE ~0.63 staff, K-Fold CV (k=5)
- **Model 6**: R² ~0.98, MAE ~2.8 shops, 24 Lebanese areas analyzed
- **Models 1 & 5**: Analytics-based (rule-based with lift/confidence)

Run `python3 scripts/evaluate_models.py` for full metric breakdowns.

## Data Limitations & Retraining

The current models are trained on a limited dataset spanning approximately 5 months of operations (August 2025 – December 2025) across only 4 branches. This constraint affects accuracy — particularly for demand forecasting (Model 2) and staffing prediction (Model 4), where small sample sizes lead to higher error margins (e.g., MAPE up to 29% for newer branches with fewer data points).

However, **the system is designed for easy retraining**. All models read from standardized CSV files in `data/cleaned/`, use centralized hyperparameters in `src/config.py`, and log experiments via MLflow. As Conut accumulates more transaction data over the coming months, retraining any model is as simple as replacing the CSV files and re-running the pipeline — no code changes required. With 12+ months of data, we expect significant accuracy improvements across all models, particularly in demand forecasting and staffing optimization.

## Engineering Highlights

- **Full-stack architecture**: FastAPI backend + Claude tool-use + responsive chat frontend
- **Centralized config** (`config.py`): All hyperparameters in one place
- **MLflow experiment tracking**: Tuning logged for Models 2, 3, 4, 6
- **Cross-validation**: LOO-CV (Model 2), K-Fold (Model 4)
- **Graceful degradation**: Works with or without mlflow/sklearn
- **Input validation**: `validate_dataframe()` on every data load
- **61 unit tests**: Full coverage across all models
- **External data research**: Model 6 uses curated Lebanese area data (CAS, WPR, Yelleb)
- **OpenClaw compatible**: Skill definitions in `.claude/skills/`

---

**Built during a 12-hour hackathon** | February 2026
