"""
Module 1: Combo Optimization
Uses Market Basket Analysis (Apriori) to find products frequently
bought together, then suggests profitable bundles.

Pure Python + pandas + numpy implementation (no mlxtend dependency).

Data inputs:
- sales_detail_(502).csv: Transaction-level data (person = basket, description = product)
- cleaned_sales_by_item_(191).csv: Product revenue data for enrichment

Key outputs:
- Frequent itemsets with support scores
- Association rules with confidence, lift, and conviction
- Ranked combo/bundle recommendations with estimated revenue impact
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
from config import validate_dataframe, get_logger, RANDOM_SEED, COMBO

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)
logger = get_logger(__name__)

# ── Items to exclude from market basket analysis ──
# These are modifiers, toppings, delivery fees, or non-product entries
EXCLUDE_ITEMS = {
    "DELIVERY CHARGE", "WATER",
}

# Items ending with these suffixes are toppings/sauces/modifiers
MODIFIER_SUFFIXES = (".", "(P)", "(R)")


def is_modifier(item_name):
    """Check if an item is a modifier/topping rather than a core product."""
    name = item_name.strip()
    if name in EXCLUDE_ITEMS:
        return True
    # Items ending with . or (P) or (R) are modifiers/sauces/toppings
    for suffix in MODIFIER_SUFFIXES:
        if name.endswith(suffix):
            return True
    # Low-price add-ons (sauces, spreads as modifiers)
    modifier_keywords = [
        "SAUCE", "SPREAD", "DIP", "SYRUP", "WHIPPED CREAM",
        "CHOCOLATE CHIPS", "RASPBERRIES", "BLUEBERRIES",
        "CRUSHED LOTUS", "BROWNIES", "STRAWBERRY.",
    ]
    for kw in modifier_keywords:
        if kw in name.upper():
            return True
    return False


def load_and_prepare_data(
    detail_path="data/cleaned/sales_detail_(502).csv",
    item_path="data/cleaned/cleaned_sales_by_item_(191).csv",
):
    """
    Load both data files and prepare them for combo analysis.

    Parameters
    ----------
    detail_path : str
        Path to sales_detail_(502).csv with columns:
        [branch, person, from_date, to_date, description, qty, price, total]
    item_path : str
        Path to cleaned_sales_by_item_(191).csv with columns:
        [division, category_name, channel, revenue]

    Returns
    -------
    dict with keys:
        'baskets_df' : DataFrame ready for basket analysis
        'product_info' : DataFrame with product revenue/category info from 191
        'raw_detail' : original 502 data
    """
    # ── Load 502 (transaction-level) ──
    detail = pd.read_csv(detail_path)
    validate_dataframe(detail, ['branch', 'person', 'description', 'qty', 'price', 'total'], "sales_detail_(502)")

    # Filter: positive qty only (exclude refunds/voids), exclude modifiers
    baskets_df = detail[
        (detail["qty"] > 0) &
        (~detail["description"].apply(is_modifier))
    ].copy()

    # Rename to standard columns for the analysis functions
    baskets_df = baskets_df.rename(columns={
        "person": "transaction_id",
        "description": "product",
        "price": "unit_price",
        "branch": "branch",
    })

    # ── Load 191 (product revenue summary) ──
    item_df = pd.read_csv(item_path)
    validate_dataframe(item_df, ['division', 'category_name', 'channel', 'revenue'], "cleaned_sales_by_item_(191)")
    
    item_df = item_df.rename(columns={
        "category_name": "product",
        "channel": "branch",
    })

    # Build product info: aggregate revenue across branches, keep category
    product_info = item_df.groupby("product").agg(
        total_revenue=("revenue", "sum"),
        division=("division", "first"),
    ).reset_index()

    logger.info(f"Loaded {len(detail)} raw transaction rows from 502")
    logger.info(f"After filtering: {len(baskets_df)} product rows (excl. modifiers/refunds)")
    logger.info(f"Unique customers (baskets): {baskets_df['transaction_id'].nunique()}")
    logger.info(f"Unique products in baskets: {baskets_df['product'].nunique()}")
    logger.info(f"Loaded {len(item_df)} product-revenue rows from 191")

    return {
        "baskets_df": baskets_df,
        "product_info": product_info,
        "raw_detail": detail,
    }


def prepare_baskets(transactions_df):
    """
    Convert transaction-level data into baskets (sets of products per transaction).

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Must contain columns: ['transaction_id', 'product']

    Returns
    -------
    list of frozenset
        Each frozenset is one transaction's set of products.
    """
    baskets = (
        transactions_df.groupby("transaction_id")["product"]
        .apply(lambda x: frozenset(x.unique()))
        .tolist()
    )
    # Only keep baskets with 2+ items (single-item baskets can't form combos)
    baskets = [b for b in baskets if len(b) >= 2]
    return baskets


def compute_support(baskets, itemset):
    """Compute support for a given itemset."""
    count = sum(1 for b in baskets if itemset.issubset(b))
    return count / len(baskets)


def apriori_frequent_itemsets(baskets, min_support=0.01, max_len=3):
    """
    Find frequent itemsets using the Apriori algorithm (pure Python).

    Parameters
    ----------
    baskets : list of frozenset
    min_support : float
    max_len : int

    Returns
    -------
    pd.DataFrame
        Columns: ['itemsets', 'support', 'items', 'itemset_size']
    """
    n = len(baskets)

    # Step 1: Count single items
    item_counts = Counter()
    for b in baskets:
        for item in b:
            item_counts[item] += 1

    # Filter by min support
    freq_1 = {frozenset([item]): count / n
              for item, count in item_counts.items()
              if count / n >= min_support}

    all_frequent = dict(freq_1)
    current_freq = set(freq_1.keys())

    # Step 2: Generate candidates of increasing size
    for k in range(2, max_len + 1):
        # Get all individual items from current frequent sets
        items = set()
        for s in current_freq:
            items.update(s)

        # Generate candidates
        candidates = set()
        for combo in combinations(sorted(items), k):
            candidate = frozenset(combo)
            # Pruning: all (k-1) subsets must be frequent
            subsets_freq = all(
                frozenset(sub) in all_frequent
                for sub in combinations(candidate, k - 1)
            )
            if subsets_freq:
                candidates.add(candidate)

        # Count support for candidates
        next_freq = {}
        for candidate in candidates:
            count = sum(1 for b in baskets if candidate.issubset(b))
            sup = count / n
            if sup >= min_support:
                next_freq[candidate] = sup

        if not next_freq:
            break

        all_frequent.update(next_freq)
        current_freq = set(next_freq.keys())

    # Build output DataFrame
    records = []
    for itemset, support in all_frequent.items():
        records.append({
            "itemsets": itemset,
            "support": round(support, 6),
            "items": sorted(list(itemset)),
            "itemset_size": len(itemset),
        })

    df = pd.DataFrame(records).sort_values("support", ascending=False).reset_index(drop=True)
    return df


def generate_rules(freq_itemsets_df, all_itemsets_df=None, min_confidence=0.1, min_lift=1.0):
    """
    Generate association rules from frequent itemsets.

    Parameters
    ----------
    freq_itemsets_df : pd.DataFrame
        Frequent itemsets (can be multi-item only, or all).
    all_itemsets_df : pd.DataFrame or None
        ALL frequent itemsets including singles, for support lookups.
        If None, uses freq_itemsets_df (assumes it contains singles too).
    min_confidence : float
    min_lift : float

    Returns
    -------
    pd.DataFrame
        Association rules with confidence, lift, support.
    """
    # Build a lookup dict from ALL itemsets (including singles)
    lookup_df = all_itemsets_df if all_itemsets_df is not None else freq_itemsets_df
    support_map = {}
    for _, row in lookup_df.iterrows():
        support_map[row["itemsets"]] = row["support"]

    rules = []
    multi = freq_itemsets_df[freq_itemsets_df["itemset_size"] >= 2]

    for _, row in multi.iterrows():
        itemset = row["itemsets"]
        sup_ab = row["support"]

        # Generate all non-empty proper subsets as antecedents
        items = list(itemset)
        for i in range(1, len(items)):
            for antecedent_tuple in combinations(items, i):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                sup_a = support_map.get(antecedent, 0)
                sup_b = support_map.get(consequent, 0)

                if sup_a == 0 or sup_b == 0:
                    continue

                confidence = sup_ab / sup_a
                lift = confidence / sup_b
                conviction = (1 - sup_b) / max(1 - confidence, 1e-10)

                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        "antecedents": antecedent,
                        "consequents": consequent,
                        "antecedent_items": sorted(list(antecedent)),
                        "consequent_items": sorted(list(consequent)),
                        "support": round(sup_ab, 6),
                        "confidence": round(confidence, 4),
                        "lift": round(lift, 4),
                        "conviction": round(conviction, 4),
                    })

    rules_df = pd.DataFrame(rules)
    if len(rules_df) > 0:
        rules_df = rules_df.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules_df


def recommend_combos(rules_df, transactions_df, product_info=None, top_n=10):
    """
    Rank and recommend product combos based on rules, with revenue estimation.

    Uses actual average prices from 502 data, enriched with 191 revenue data.
    """
    if len(rules_df) == 0:
        return pd.DataFrame()

    # Get average prices from transaction data
    avg_prices = transactions_df.groupby("product")["unit_price"].mean().to_dict()

    # Enrich with 191 revenue data if available
    revenue_map = {}
    division_map = {}
    if product_info is not None:
        for _, row in product_info.iterrows():
            revenue_map[row["product"]] = row["total_revenue"]
            division_map[row["product"]] = row["division"]

    combos = []
    seen = set()

    for _, rule in rules_df.iterrows():
        combo_items = sorted(list(rule["antecedents"]) + list(rule["consequents"]))
        combo_key = tuple(combo_items)

        if combo_key in seen:
            continue
        seen.add(combo_key)

        full_price = sum(avg_prices.get(item, 0) for item in combo_items)
        discount_pct = 0.10 + 0.05 * (len(combo_items) - 2) if len(combo_items) > 2 else 0.10
        combo_price = round(full_price * (1 - discount_pct), 2)
        savings = round(full_price - combo_price, 2)

        # Revenue potential from 191 data
        total_revenue = sum(revenue_map.get(item, 0) for item in combo_items)

        # Categories from 191
        categories = list(set(
            division_map.get(item, "Unknown") for item in combo_items
        ))

        score = round(
            rule["lift"] * 0.4 +
            rule["confidence"] * 0.3 * 10 +
            rule["support"] * 0.3 * 100,
            3
        )

        combos.append({
            "combo_items": combo_items,
            "combo_size": len(combo_items),
            "full_price": round(full_price, 2),
            "combo_price": combo_price,
            "savings": savings,
            "discount_pct": round(discount_pct * 100, 1),
            "total_revenue_191": round(total_revenue, 2),
            "categories": categories,
            "support": round(rule["support"], 4),
            "confidence": round(rule["confidence"], 4),
            "lift": round(rule["lift"], 4),
            "score": score,
        })

    combos_df = pd.DataFrame(combos).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    combos_df.index = combos_df.index + 1
    combos_df.index.name = "rank"
    return combos_df


def analyze_combos_by_branch(transactions_df, product_info=None, min_support=0.01, top_n=5):
    """Run combo analysis per branch."""
    results = {}
    for branch in transactions_df["branch"].unique():
        branch_txns = transactions_df[transactions_df["branch"] == branch]
        try:
            baskets = prepare_baskets(branch_txns)
            if len(baskets) < 3:
                logger.info(f"  → {branch}: Only {len(baskets)} multi-item baskets, skipping")
                results[branch] = pd.DataFrame()
                continue
            freq = apriori_frequent_itemsets(baskets, min_support=min_support)
            freq_multi = freq[freq["itemset_size"] >= 2]
            if len(freq_multi) == 0:
                results[branch] = pd.DataFrame()
                continue
            rules = generate_rules(freq_multi, all_itemsets_df=freq)
            if len(rules) == 0:
                results[branch] = pd.DataFrame()
                continue
            combos = recommend_combos(rules, branch_txns, product_info=product_info, top_n=top_n)
            results[branch] = combos
        except Exception as e:
            results[branch] = pd.DataFrame()
            logger.warning(f"Could not analyze combos for {branch}: {e}")
    return results


def run_full_combo_analysis(
    detail_path="data/cleaned/sales_detail_(502).csv",
    item_path="data/cleaned/cleaned_sales_by_item_(191).csv",
    min_support=0.01,
    top_n=10,
):
    """
    Complete combo optimization pipeline using both cleaned data files.

    Parameters
    ----------
    detail_path : str
        Path to the 502 transaction detail CSV
    item_path : str
        Path to the 191 sales-by-item CSV
    min_support : float
        Minimum support threshold for Apriori
    top_n : int
        Number of top combos to return

    Returns
    -------
    dict with keys:
        'frequent_itemsets', 'rules', 'recommended_combos',
        'branch_combos', 'summary', 'data'
    """
    logger.info("=" * 60)
    logger.info("COMBO OPTIMIZATION ANALYSIS")
    logger.info("=" * 60)

    logger.info("\n[1/6] Loading and preparing data...")
    data = load_and_prepare_data(detail_path, item_path)
    baskets_df = data["baskets_df"]
    product_info = data["product_info"]

    logger.info("\n[2/6] Preparing baskets...")
    baskets = prepare_baskets(baskets_df)
    n_transactions = len(baskets)
    all_products = set()
    for b in baskets:
        all_products.update(b)
    logger.info(f"  → {n_transactions} multi-item baskets, {len(all_products)} unique products")

    logger.info("\n[3/6] Finding frequent itemsets (Apriori)...")
    freq_items = apriori_frequent_itemsets(baskets, min_support=min_support)
    freq_multi = freq_items[freq_items["itemset_size"] >= 2]
    logger.info(f"  → {len(freq_items)} total itemsets, {len(freq_multi)} multi-item sets")

    logger.info("\n[4/6] Generating association rules...")
    if len(freq_multi) > 0:
        rules = generate_rules(freq_multi, all_itemsets_df=freq_items)
        logger.info(f"  → {len(rules)} rules found")
    else:
        rules = pd.DataFrame()
        logger.info("  → No multi-item frequent itemsets found. Try lowering min_support.")

    logger.info("\n[5/6] Ranking combo recommendations...")
    if len(rules) > 0:
        combos = recommend_combos(rules, baskets_df, product_info=product_info, top_n=top_n)
        logger.info(f"  → Top {len(combos)} combos identified")
    else:
        combos = pd.DataFrame()

    logger.info("\n[6/6] Running per-branch analysis...")
    branch_combos = analyze_combos_by_branch(
        baskets_df, product_info=product_info, min_support=min_support
    )
    for branch, df in branch_combos.items():
        logger.info(f"  → {branch}: {len(df)} combos")

    summary = _build_summary(combos, rules, n_transactions, freq_items)
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)

    return {
        "frequent_itemsets": freq_items,
        "rules": rules,
        "recommended_combos": combos,
        "branch_combos": branch_combos,
        "summary": summary,
        "data": data,
    }


def _build_summary(combos, rules, n_transactions, freq_items):
    lines = ["COMBO OPTIMIZATION SUMMARY", "=" * 40]
    if len(combos) > 0:
        top = combos.iloc[0]
        lines.append(f"\nTop Recommended Combo: {' + '.join(top['combo_items'])}")
        lines.append(f"  Bundle Price: {top['combo_price']:,.0f} LBP (save {top['savings']:,.0f} LBP, {top['discount_pct']}% off)")
        lines.append(f"  Lift: {top['lift']}x")
        lines.append(f"  Confidence: {top['confidence']*100:.1f}%")
        if top.get("total_revenue_191", 0) > 0:
            lines.append(f"  Combined Revenue (from 191): {top['total_revenue_191']:,.0f} LBP")
            lines.append(f"  Categories: {', '.join(top['categories'])}")
        lines.append(f"\nTotal combos analyzed from {n_transactions} multi-item baskets")
        lines.append(f"Frequent itemsets: {len(freq_items)}")
        lines.append(f"Association rules: {len(rules)}")
    else:
        lines.append("\nNo strong combos found with current thresholds.")
        lines.append("Recommendation: Lower min_support or review data quality.")
    return "\n".join(lines)


# === Agent-callable function for OpenClaw integration ===

def get_combo_recommendations(
    branch=None,
    min_support=0.01,
    top_n=5,
    detail_path="data/cleaned/sales_detail_(502).csv",
    item_path="data/cleaned/cleaned_sales_by_item_(191).csv",
):
    """
    OpenClaw-compatible function: Returns best product combos to promote.
    Uses both 502 (transaction baskets) and 191 (product revenue) data.
    """
    try:
        data = load_and_prepare_data(detail_path, item_path)
        baskets_df = data["baskets_df"]
        product_info = data["product_info"]

        if branch:
            baskets_df = baskets_df[baskets_df["branch"] == branch]
            if len(baskets_df) == 0:
                return {"status": "error", "message": f"No data found for branch '{branch}'"}

        baskets = prepare_baskets(baskets_df)
        if len(baskets) < 3:
            return {"status": "error", "message": f"Only {len(baskets)} multi-item baskets. Need more data."}

        freq_items = apriori_frequent_itemsets(baskets, min_support=min_support)
        freq_multi = freq_items[freq_items["itemset_size"] >= 2]

        if len(freq_multi) == 0:
            return {"status": "error", "message": "No frequent item pairs found. Try lowering min_support."}

        rules = generate_rules(freq_multi, all_itemsets_df=freq_items)
        if len(rules) == 0:
            return {"status": "error", "message": "No strong association rules found."}

        combos = recommend_combos(rules, baskets_df, product_info=product_info, top_n=top_n)

        combos_list = []
        for _, row in combos.iterrows():
            combos_list.append({
                "items": row["combo_items"],
                "combo_price": row["combo_price"],
                "savings": row["savings"],
                "lift": row["lift"],
                "confidence": row["confidence"],
                "total_revenue": row.get("total_revenue_191", 0),
                "categories": row.get("categories", []),
            })

        return {
            "status": "success",
            "branch": branch or "all",
            "combos": combos_list,
            "total_rules": len(rules),
            "total_baskets": len(baskets),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
