"""
Synthetic data generator for development and testing.
Generates realistic placeholder data matching expected Conut bakery schemas.
Replace with real cleaned data when available from Phase 1.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

BRANCHES = ["Branch_A", "Branch_B", "Branch_C", "Branch_D", "Branch_E"]
PRODUCTS = [
    # Bakery
    "Croissant", "Chocolate Cake", "Cheese Danish", "Muffin", "Donut",
    "Cinnamon Roll", "Bagel", "Brownie", "Cookie", "Scone",
    # Coffee
    "Espresso", "Latte", "Cappuccino", "Americano", "Mocha",
    "Cold Brew", "Iced Coffee", "Macchiato",
    # Milkshake
    "Chocolate Milkshake", "Vanilla Milkshake", "Strawberry Milkshake",
    "Oreo Milkshake", "Caramel Milkshake",
    # Beverages
    "Fresh Orange Juice", "Iced Tea", "Lemonade", "Smoothie",
]

PRODUCT_CATEGORIES = {
    "Croissant": "Bakery", "Chocolate Cake": "Bakery", "Cheese Danish": "Bakery",
    "Muffin": "Bakery", "Donut": "Bakery", "Cinnamon Roll": "Bakery",
    "Bagel": "Bakery", "Brownie": "Bakery", "Cookie": "Bakery", "Scone": "Bakery",
    "Espresso": "Coffee", "Latte": "Coffee", "Cappuccino": "Coffee",
    "Americano": "Coffee", "Mocha": "Coffee", "Cold Brew": "Coffee",
    "Iced Coffee": "Coffee", "Macchiato": "Coffee",
    "Chocolate Milkshake": "Milkshake", "Vanilla Milkshake": "Milkshake",
    "Strawberry Milkshake": "Milkshake", "Oreo Milkshake": "Milkshake",
    "Caramel Milkshake": "Milkshake",
    "Fresh Orange Juice": "Beverage", "Iced Tea": "Beverage",
    "Lemonade": "Beverage", "Smoothie": "Beverage",
}

PRODUCT_PRICES = {
    "Croissant": 3.50, "Chocolate Cake": 5.00, "Cheese Danish": 3.75,
    "Muffin": 2.50, "Donut": 2.00, "Cinnamon Roll": 3.25,
    "Bagel": 2.75, "Brownie": 3.00, "Cookie": 1.50, "Scone": 2.50,
    "Espresso": 2.50, "Latte": 4.50, "Cappuccino": 4.00,
    "Americano": 3.00, "Mocha": 4.75, "Cold Brew": 3.75,
    "Iced Coffee": 3.50, "Macchiato": 4.25,
    "Chocolate Milkshake": 5.50, "Vanilla Milkshake": 5.00,
    "Strawberry Milkshake": 5.50, "Oreo Milkshake": 6.00,
    "Caramel Milkshake": 5.75,
    "Fresh Orange Juice": 4.00, "Iced Tea": 2.50,
    "Lemonade": 3.00, "Smoothie": 5.00,
}

# Common item pairings (for realistic basket generation)
COMMON_PAIRS = [
    ("Croissant", "Latte"), ("Croissant", "Cappuccino"),
    ("Muffin", "Americano"), ("Bagel", "Espresso"),
    ("Chocolate Cake", "Latte"), ("Donut", "Iced Coffee"),
    ("Cookie", "Chocolate Milkshake"), ("Brownie", "Vanilla Milkshake"),
    ("Scone", "Cold Brew"), ("Cinnamon Roll", "Mocha"),
    ("Cheese Danish", "Cappuccino"), ("Muffin", "Fresh Orange Juice"),
]

SHIFTS = ["Morning", "Afternoon", "Evening"]


def generate_transactions(n_transactions=5000, start_date="2024-01-01", end_date="2025-12-31"):
    """
    Generate synthetic transaction-level data.
    Each row = one item in a transaction (basket).
    """
    random.seed(42)
    np.random.seed(42)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    date_range = (end - start).days

    records = []
    txn_id = 1000

    for _ in range(n_transactions):
        txn_id += 1
        date = start + timedelta(days=random.randint(0, date_range))
        branch = random.choice(BRANCHES)
        hour = np.random.choice(range(6, 22), p=_hour_distribution())
        shift = "Morning" if hour < 12 else ("Afternoon" if hour < 17 else "Evening")

        # Generate basket: 1-5 items with realistic pairing bias
        basket_size = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.35, 0.30, 0.15, 0.05])
        basket = _generate_basket(basket_size)

        for item in basket:
            qty = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
            records.append({
                "transaction_id": f"TXN-{txn_id}",
                "date": date,
                "hour": hour,
                "shift": shift,
                "branch": branch,
                "product": item,
                "category": PRODUCT_CATEGORIES[item],
                "quantity": qty,
                "unit_price": PRODUCT_PRICES[item],
                "total_price": round(PRODUCT_PRICES[item] * qty, 2),
            })

    return pd.DataFrame(records)


def generate_daily_branch_sales(transactions_df=None):
    """
    Generate daily aggregated sales by branch.
    If transactions_df is provided, aggregates from it. Otherwise generates synthetic.
    """
    if transactions_df is not None:
        daily = transactions_df.groupby(["date", "branch"]).agg(
            total_revenue=("total_price", "sum"),
            total_transactions=("transaction_id", "nunique"),
            total_items_sold=("quantity", "sum"),
        ).reset_index()
        daily["avg_transaction_value"] = round(daily["total_revenue"] / daily["total_transactions"], 2)
        return daily

    # Synthetic fallback
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    records = []
    for date in dates:
        for branch in BRANCHES:
            base_revenue = {"Branch_A": 1200, "Branch_B": 900, "Branch_C": 1500,
                            "Branch_D": 700, "Branch_E": 1100}[branch]
            # Add seasonality and day-of-week effects
            dow_factor = 1.3 if date.dayofweek >= 5 else 1.0
            month_factor = 1 + 0.15 * np.sin(2 * np.pi * date.month / 12)
            noise = np.random.normal(1.0, 0.1)
            revenue = round(base_revenue * dow_factor * month_factor * noise, 2)
            txn_count = max(int(revenue / np.random.uniform(12, 20)), 1)
            items = int(txn_count * np.random.uniform(1.8, 3.2))
            records.append({
                "date": date,
                "branch": branch,
                "total_revenue": revenue,
                "total_transactions": txn_count,
                "total_items_sold": items,
                "avg_transaction_value": round(revenue / txn_count, 2),
            })
    return pd.DataFrame(records)


def generate_attendance_data(start_date="2024-01-01", end_date="2025-12-31"):
    """
    Generate synthetic employee attendance / shift data.
    """
    np.random.seed(42)
    dates = pd.date_range(start_date, end_date, freq="D")
    records = []
    employee_id = 100

    for branch in BRANCHES:
        base_staff = {"Branch_A": 8, "Branch_B": 6, "Branch_C": 10,
                      "Branch_D": 5, "Branch_E": 7}[branch]
        for date in dates:
            for shift in SHIFTS:
                shift_factor = {"Morning": 1.0, "Afternoon": 0.8, "Evening": 0.6}[shift]
                dow_factor = 1.2 if date.dayofweek >= 5 else 1.0
                n_staff = max(int(base_staff * shift_factor * dow_factor + np.random.normal(0, 0.5)), 1)
                for _ in range(n_staff):
                    employee_id += 1
                    hours_worked = round(np.random.uniform(4, 8), 1)
                    records.append({
                        "date": date,
                        "branch": branch,
                        "shift": shift,
                        "employee_id": f"EMP-{employee_id % 50 + 1:03d}",
                        "hours_worked": hours_worked,
                    })
    return pd.DataFrame(records)


def generate_branch_performance():
    """
    Generate branch-level KPI summary data for expansion feasibility analysis.
    """
    np.random.seed(42)
    records = []
    for branch in BRANCHES:
        monthly_rev = {"Branch_A": 36000, "Branch_B": 27000, "Branch_C": 45000,
                       "Branch_D": 21000, "Branch_E": 33000}[branch]
        records.append({
            "branch": branch,
            "monthly_avg_revenue": monthly_rev,
            "monthly_avg_customers": int(monthly_rev / np.random.uniform(12, 18)),
            "avg_ticket_size": round(monthly_rev / int(monthly_rev / np.random.uniform(12, 18)), 2),
            "yoy_growth_pct": round(np.random.uniform(-5, 25), 1),
            "rent_cost": int(np.random.uniform(4000, 10000)),
            "staff_cost": int(np.random.uniform(8000, 20000)),
            "cogs_pct": round(np.random.uniform(25, 40), 1),
            "profit_margin_pct": round(np.random.uniform(8, 25), 1),
            "avg_daily_footfall": int(monthly_rev / 30 / np.random.uniform(10, 16)),
            "customer_satisfaction_score": round(np.random.uniform(3.5, 4.8), 1),
            "area_population_density": int(np.random.uniform(5000, 25000)),
            "nearby_competitors": np.random.randint(1, 8),
        })
    return pd.DataFrame(records)


def _hour_distribution():
    """Realistic hourly distribution - peaks at morning and lunch."""
    raw = [0.02, 0.05, 0.10, 0.14, 0.12, 0.08, 0.08, 0.07,  # 6-13
           0.06, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03, 0.03]  # 14-21
    total = sum(raw)
    return [x / total for x in raw]


def _generate_basket(size):
    """Generate a realistic basket of items with pairing bias."""
    if size == 1:
        return [random.choice(PRODUCTS)]

    basket = set()
    # Start with a common pair with 60% probability
    if random.random() < 0.6 and size >= 2:
        pair = random.choice(COMMON_PAIRS)
        basket.update(pair)

    while len(basket) < size:
        basket.add(random.choice(PRODUCTS))

    return list(basket)[:size]


if __name__ == "__main__":
    print("Generating synthetic data...")
    txns = generate_transactions()
    print(f"Transactions: {len(txns)} rows, {txns['transaction_id'].nunique()} unique transactions")

    daily = generate_daily_branch_sales(txns)
    print(f"Daily branch sales: {len(daily)} rows")

    attendance = generate_attendance_data()
    print(f"Attendance: {len(attendance)} rows")

    perf = generate_branch_performance()
    print(f"Branch performance: {len(perf)} rows")
