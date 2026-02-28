"""
Module 5: Coffee & Milkshake Growth Strategy
Develops data-driven strategies to increase coffee and milkshake sales
using three data sources:

- cleaned_sales_by_item_(191).csv: Product-level revenue by branch
- cleaned_revenue_summary_(136).csv: Channel breakdown (delivery/takeaway/table) per division
- sales_detail_(502).csv: Customer-level transactions for cross-sell analysis

Key outputs:
- Category performance analysis by branch
- Channel opportunity identification (delivery, takeaway, table)
- Product-level rankings and underperformers
- Cross-sell patterns (what do coffee/shake buyers also order?)
- Actionable growth strategies with revenue estimates
"""

import pandas as pd
import numpy as np
from config import validate_dataframe, get_logger, RANDOM_SEED, COFFEE

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)
logger = get_logger(__name__)

# ── Category definitions ──
COFFEE_DIVISIONS = ["Hot-Coffee Based", "Frappes"]
MILKSHAKE_DIVISIONS = ["Shakes"]
DRINKS_DIVISIONS = ["Hot and Cold Drinks"]

COFFEE_KEYWORDS = [
    "COFFEE", "CAFFE", "ESPRESSO", "AMERICANO", "LATTE", "MOCHA",
    "CAPPUCCINO", "MACCHIATO", "AFFOGATO", "FRAPPE",
]
MILKSHAKE_KEYWORDS = ["MILKSHAKE"]


def load_data(
    item_path="data/cleaned/cleaned_sales_by_item_(191).csv",
    revenue_path="data/cleaned/cleaned_revenue_summary_(136).csv",
    detail_path="data/cleaned/sales_detail_(502).csv",
):
    """Load all three data sources."""
    df191 = pd.read_csv(item_path)
    validate_dataframe(df191, ['division', 'category_name', 'channel', 'revenue'], "cleaned_sales_by_item_(191)")
    
    df136 = pd.read_csv(revenue_path)
    validate_dataframe(df136, ['division', 'category', 'delivery', 'takeaway', 'table', 'total'], "cleaned_revenue_summary_(136)")
    
    df502 = pd.read_csv(detail_path)
    validate_dataframe(df502, ['branch', 'person', 'description'], "sales_detail_(502)")

    logger.info(f"Loaded 191: {len(df191)} rows (product revenue by branch)")
    logger.info(f"Loaded 136: {len(df136)} rows (channel breakdown by division)")
    logger.info(f"Loaded 502: {len(df502)} rows (customer transactions)")

    return df191, df136, df502


def analyze_category_performance(df191):
    """
    Analyze coffee and milkshake performance by branch from 191 data.
    Returns detailed product-level and branch-level breakdowns.
    """
    results = {}

    for category_name, divisions in [("Coffee", COFFEE_DIVISIONS), ("Milkshake", MILKSHAKE_DIVISIONS)]:
        cat_data = df191[df191["division"].isin(divisions)].copy()

        # Branch breakdown
        branch_summary = cat_data.groupby("channel").agg(
            total_revenue=("revenue", "sum"),
            product_count=("category_name", "nunique"),
            avg_product_revenue=("revenue", "mean"),
            top_product_revenue=("revenue", "max"),
        ).reset_index().rename(columns={"channel": "branch"})

        branch_summary["revenue_share_pct"] = round(
            branch_summary["total_revenue"] / branch_summary["total_revenue"].sum() * 100, 1
        )
        branch_summary = branch_summary.sort_values("total_revenue", ascending=False)

        # Product rankings (across all branches)
        product_summary = cat_data.groupby("category_name").agg(
            total_revenue=("revenue", "sum"),
            branches_present=("channel", "nunique"),
            avg_revenue_per_branch=("revenue", "mean"),
        ).reset_index().sort_values("total_revenue", ascending=False)

        product_summary["revenue_share_pct"] = round(
            product_summary["total_revenue"] / product_summary["total_revenue"].sum() * 100, 1
        )

        # Identify underperformers (below average, present in fewer branches)
        avg_rev = product_summary["total_revenue"].mean()
        product_summary["is_underperformer"] = (
            (product_summary["total_revenue"] < avg_rev * 0.5) &
            (product_summary["branches_present"] < 4)
        )

        # Branch gaps — products not available everywhere
        all_branches = cat_data["channel"].unique()
        product_branch_matrix = cat_data.pivot_table(
            index="category_name", columns="channel", values="revenue", aggfunc="sum", fill_value=0
        )
        gaps = []
        for product in product_branch_matrix.index:
            missing = [b for b in all_branches if product_branch_matrix.loc[product, b] == 0]
            if missing and product_branch_matrix.loc[product].sum() > avg_rev * 0.3:
                gaps.append({
                    "product": product,
                    "missing_branches": missing,
                    "current_revenue": product_branch_matrix.loc[product].sum(),
                })

        results[category_name] = {
            "branch_summary": branch_summary,
            "product_summary": product_summary,
            "total_revenue": cat_data["revenue"].sum(),
            "total_products": cat_data["category_name"].nunique(),
            "branch_gaps": gaps,
        }

    return results


def analyze_channel_opportunities(df136):
    """
    Analyze delivery/takeaway/table split for coffee and milkshake divisions
    from 136 data. Identifies channel growth opportunities.
    """
    relevant_divs = COFFEE_DIVISIONS + MILKSHAKE_DIVISIONS + DRINKS_DIVISIONS
    channel_data = df136[df136["category"].isin(relevant_divs)].copy()

    results = {}

    for category_name, divisions in [("Coffee", COFFEE_DIVISIONS), ("Milkshake", MILKSHAKE_DIVISIONS)]:
        cat_channels = channel_data[channel_data["category"].isin(divisions)].copy()

        branch_channels = []
        for _, row in cat_channels.iterrows():
            total = row["total"]
            if total <= 0:
                continue
            branch_channels.append({
                "branch": row["division"],
                "category": row["category"],
                "delivery": row["delivery"],
                "takeaway": row["takeaway"],
                "table": row["table"],
                "total": total,
                "delivery_pct": round(row["delivery"] / total * 100, 1) if total > 0 else 0,
                "takeaway_pct": round(row["takeaway"] / total * 100, 1) if total > 0 else 0,
                "table_pct": round(row["table"] / total * 100, 1) if total > 0 else 0,
            })

        branch_channels_df = pd.DataFrame(branch_channels)

        # Identify opportunities
        opportunities = []
        for _, row in branch_channels_df.iterrows():
            if row["delivery_pct"] == 0 and row["total"] > 10_000_000:
                opportunities.append({
                    "branch": row["branch"],
                    "category": row["category"],
                    "opportunity": "No delivery channel",
                    "potential": f"Other branches generate delivery revenue — estimated {row['total'] * 0.05:,.0f} LBP potential",
                })
            if row["takeaway_pct"] == 0 and row["total"] > 10_000_000:
                opportunities.append({
                    "branch": row["branch"],
                    "category": row["category"],
                    "opportunity": "No takeaway channel",
                    "potential": f"Estimated {row['total'] * 0.1:,.0f} LBP potential from takeaway",
                })

        results[category_name] = {
            "channel_breakdown": branch_channels_df,
            "opportunities": opportunities,
        }

    return results


def analyze_cross_sell_patterns(df502):
    """
    Analyze what other products coffee/milkshake buyers also purchase
    from 502 transaction data. Identifies cross-sell opportunities.
    """
    results = {}

    for category_name, keywords in [("Coffee", COFFEE_KEYWORDS), ("Milkshake", MILKSHAKE_KEYWORDS)]:
        # Find customers who bought coffee/milkshakes
        mask = df502["description"].str.upper().apply(
            lambda x: any(kw in str(x) for kw in keywords)
        )
        category_buyers = df502[mask]["person"].unique()

        if len(category_buyers) == 0:
            results[category_name] = {
                "buyer_count": 0,
                "cross_sell_products": pd.DataFrame(),
                "avg_basket_value": 0,
            }
            continue

        # Get all items these buyers purchased
        buyer_baskets = df502[df502["person"].isin(category_buyers)].copy()
        non_category_items = buyer_baskets[~mask]

        # Filter out non-products from cross-sell
        exclude = {"WATER", "DELIVERY CHARGE"}
        non_category_items = non_category_items[~non_category_items["description"].isin(exclude)]

        # What else do they buy?
        if len(non_category_items) > 0:
            cross_sell = non_category_items.groupby("description").agg(
                frequency=("person", "nunique"),
                total_revenue=("price", lambda x: x[x > 0].sum()),
            ).reset_index()
            cross_sell["pct_of_buyers"] = round(
                cross_sell["frequency"] / len(category_buyers) * 100, 1
            )
            cross_sell = cross_sell.sort_values("frequency", ascending=False)
        else:
            cross_sell = pd.DataFrame()

        # Average basket value for category buyers
        basket_totals = buyer_baskets.groupby("person")["total"].first()
        avg_basket = basket_totals.mean()

        # Branch breakdown of category buyers
        buyer_branches = buyer_baskets.groupby("branch")["person"].nunique().reset_index()
        buyer_branches.columns = ["branch", "buyer_count"]

        results[category_name] = {
            "buyer_count": len(category_buyers),
            "cross_sell_products": cross_sell,
            "avg_basket_value": avg_basket,
            "buyer_branches": buyer_branches,
        }

    return results


def generate_growth_strategies(perf_results, channel_results, cross_sell_results):
    """
    Synthesize all analyses into actionable growth strategies.
    """
    strategies = []

    for category in ["Coffee", "Milkshake"]:
        p = perf_results.get(category, {})
        ch = channel_results.get(category, {})
        cs = cross_sell_results.get(category, {})

        cat_strategies = []

        # Strategy 1: Branch expansion gaps
        gaps = p.get("branch_gaps", [])
        if gaps:
            for gap in gaps[:3]:
                cat_strategies.append({
                    "strategy": f"Expand {gap['product']} to {', '.join(gap['missing_branches'])}",
                    "type": "Product Expansion",
                    "rationale": f"Currently generates {gap['current_revenue']:,.0f} LBP without these branches",
                    "estimated_impact": f"{gap['current_revenue'] * 0.3:,.0f} LBP additional revenue",
                    "priority": "High",
                })

        # Strategy 2: Channel opportunities
        opps = ch.get("opportunities", [])
        for opp in opps[:3]:
            cat_strategies.append({
                "strategy": f"{opp['opportunity']} for {opp['category']} at {opp['branch']}",
                "type": "Channel Growth",
                "rationale": opp["potential"],
                "estimated_impact": opp["potential"],
                "priority": "Medium",
            })

        # Strategy 3: Cross-sell bundles
        cross_prods = cs.get("cross_sell_products", pd.DataFrame())
        if len(cross_prods) > 0:
            top_cross = cross_prods.head(3)
            for _, row in top_cross.iterrows():
                cat_strategies.append({
                    "strategy": f"Bundle {category.lower()} with {row['description']}",
                    "type": "Cross-Sell",
                    "rationale": f"{row['pct_of_buyers']}% of {category.lower()} buyers also buy this",
                    "estimated_impact": f"Increase basket value for {cs.get('buyer_count', 0)} buyers",
                    "priority": "Medium",
                })

        # Strategy 4: Underperformer action
        product_summary = p.get("product_summary", pd.DataFrame())
        if len(product_summary) > 0:
            underperformers = product_summary[product_summary["is_underperformer"] == True]
            if len(underperformers) > 0:
                cat_strategies.append({
                    "strategy": f"Review {len(underperformers)} underperforming {category.lower()} products",
                    "type": "Menu Optimization",
                    "rationale": "Products with <50% avg revenue and limited branch presence",
                    "estimated_impact": "Reduce menu complexity or boost via promotions",
                    "priority": "Low",
                })

        # Strategy 5: Top performer doubling down
        if len(product_summary) > 0:
            top = product_summary.iloc[0]
            cat_strategies.append({
                "strategy": f"Promote {top['category_name']} as flagship {category.lower()}",
                "type": "Marketing Focus",
                "rationale": f"Top seller at {top['total_revenue']:,.0f} LBP ({top['revenue_share_pct']}% of category)",
                "estimated_impact": f"10% growth = {top['total_revenue'] * 0.1:,.0f} LBP",
                "priority": "High",
            })

        strategies.append({
            "category": category,
            "strategies": cat_strategies,
            "total_revenue": p.get("total_revenue", 0),
        })

    return strategies


def run_full_coffee_milkshake_analysis(
    item_path="data/cleaned/cleaned_sales_by_item_(191).csv",
    revenue_path="data/cleaned/cleaned_revenue_summary_(136).csv",
    detail_path="data/cleaned/sales_detail_(502).csv",
):
    """
    Complete coffee & milkshake growth analysis pipeline.
    """
    logger.info("=" * 60)
    logger.info("COFFEE & MILKSHAKE GROWTH STRATEGY ANALYSIS")
    logger.info("=" * 60)

    logger.info("\n[1/5] Loading data...")
    df191, df136, df502 = load_data(item_path, revenue_path, detail_path)

    logger.info("\n[2/5] Analyzing category performance (191)...")
    perf = analyze_category_performance(df191)
    for cat, data in perf.items():
        logger.info(f"  → {cat}: {data['total_products']} products, {data['total_revenue']:,.0f} LBP total")

    logger.info("\n[3/5] Analyzing channel opportunities (136)...")
    channels = analyze_channel_opportunities(df136)
    for cat, data in channels.items():
        n_opps = len(data["opportunities"])
        logger.info(f"  → {cat}: {n_opps} channel opportunities found")

    logger.info("\n[4/5] Analyzing cross-sell patterns (502)...")
    cross_sell = analyze_cross_sell_patterns(df502)
    for cat, data in cross_sell.items():
        logger.info(f"  → {cat}: {data['buyer_count']} buyers, avg basket {data['avg_basket_value']:,.0f} LBP")

    logger.info("\n[5/5] Generating growth strategies...")
    strategies = generate_growth_strategies(perf, channels, cross_sell)
    for s in strategies:
        logger.info(f"  → {s['category']}: {len(s['strategies'])} strategies")

    summary = _build_summary(perf, channels, cross_sell, strategies)

    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)

    return {
        "performance": perf,
        "channels": channels,
        "cross_sell": cross_sell,
        "strategies": strategies,
        "summary": summary,
    }


def _build_summary(perf, channels, cross_sell, strategies):
    lines = ["COFFEE & MILKSHAKE GROWTH SUMMARY", "=" * 40]

    for cat in ["Coffee", "Milkshake"]:
        p = perf.get(cat, {})
        lines.append(f"\n--- {cat.upper()} ---")
        lines.append(f"Total Revenue: {p.get('total_revenue', 0):,.0f} LBP")
        lines.append(f"Products: {p.get('total_products', 0)}")

        prod_summary = p.get("product_summary", pd.DataFrame())
        if len(prod_summary) > 0:
            top = prod_summary.iloc[0]
            lines.append(f"Top Seller: {top['category_name']} ({top['total_revenue']:,.0f} LBP)")

        branch_summary = p.get("branch_summary", pd.DataFrame())
        if len(branch_summary) > 0:
            leader = branch_summary.iloc[0]
            lines.append(f"Strongest Branch: {leader['branch']} ({leader['revenue_share_pct']}%)")

        cs = cross_sell.get(cat, {})
        if cs.get("buyer_count", 0) > 0:
            lines.append(f"Active Buyers (502): {cs['buyer_count']}")
            lines.append(f"Avg Basket Value: {cs['avg_basket_value']:,.0f} LBP")

    return "\n".join(lines)


# === Agent-callable function for OpenClaw integration ===

def get_coffee_milkshake_analysis(
    category=None,
    branch=None,
    item_path="data/cleaned/cleaned_sales_by_item_(191).csv",
    revenue_path="data/cleaned/cleaned_revenue_summary_(136).csv",
    detail_path="data/cleaned/sales_detail_(502).csv",
):
    """
    OpenClaw-compatible function: Returns coffee/milkshake growth analysis and strategies.
    """
    try:
        df191, df136, df502 = load_data(item_path, revenue_path, detail_path)

        perf = analyze_category_performance(df191)
        channels = analyze_channel_opportunities(df136)
        cross_sell = analyze_cross_sell_patterns(df502)
        strategies = generate_growth_strategies(perf, channels, cross_sell)

        categories = ["Coffee", "Milkshake"]
        if category:
            categories = [c for c in categories if c.lower() == category.lower()]

        result = {"status": "success", "categories": []}

        for cat in categories:
            p = perf.get(cat, {})
            ch = channels.get(cat, {})
            cs = cross_sell.get(cat, {})
            strat = next((s for s in strategies if s["category"] == cat), {})

            prod_summary = p.get("product_summary", pd.DataFrame())
            branch_summary = p.get("branch_summary", pd.DataFrame())

            if branch and len(branch_summary) > 0:
                branch_summary = branch_summary[branch_summary["branch"] == branch]

            cat_result = {
                "category": cat,
                "total_revenue": p.get("total_revenue", 0),
                "total_products": p.get("total_products", 0),
                "top_products": [],
                "branch_performance": [],
                "channel_opportunities": ch.get("opportunities", []),
                "cross_sell_products": [],
                "strategies": strat.get("strategies", []),
            }

            if len(prod_summary) > 0:
                for _, row in prod_summary.head(5).iterrows():
                    cat_result["top_products"].append({
                        "product": row["category_name"],
                        "revenue": row["total_revenue"],
                        "share_pct": row["revenue_share_pct"],
                        "branches": int(row["branches_present"]),
                    })

            if len(branch_summary) > 0:
                for _, row in branch_summary.iterrows():
                    cat_result["branch_performance"].append({
                        "branch": row["branch"],
                        "revenue": row["total_revenue"],
                        "share_pct": row["revenue_share_pct"],
                        "products": int(row["product_count"]),
                    })

            cross_prods = cs.get("cross_sell_products", pd.DataFrame())
            if len(cross_prods) > 0:
                for _, row in cross_prods.head(5).iterrows():
                    cat_result["cross_sell_products"].append({
                        "product": row["description"],
                        "pct_of_buyers": row["pct_of_buyers"],
                        "frequency": int(row["frequency"]),
                    })

            cat_result["buyer_count"] = cs.get("buyer_count", 0)
            cat_result["avg_basket_value"] = cs.get("avg_basket_value", 0)

            result["categories"].append(cat_result)

        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}
