"""
Clean & preprocess rep_s_00150.csv (Omega POS Customer Delivery Orders Report)
-------------------------------------------------------------------------------
Input:  Messy multi-page POS export with repeated page headers, branch headers,
        summary rows, copyright footer, and inconsistent column layouts (page 15
        has an extra empty column shifting Total/No. of Orders).
Output: Clean CSV with columns:
        [branch, customer, address, phone, first_order, last_order, total, num_orders]
"""

import pandas as pd
import re
import csv
from io import StringIO

INPUT_PATH  = "/Users/ronniesaba/Downloads/Hackathon/Conut bakery Scaled Data /rep_s_00150.csv"  # ← update to your path
OUTPUT_PATH = "/Users/ronniesaba/Downloads/Hackathon/Cleaned_Data/cleaned_customer_orders.csv"               # ← update to your path

# ── 1. Read raw lines ────────────────────────────────────────────
with open(INPUT_PATH, "r", encoding="utf-8-sig") as f:
    raw_lines = f.readlines()

print(f"Raw lines read: {len(raw_lines)}")

# ── 2. Filter out junk rows ──────────────────────────────────────
JUNK_PATTERNS = [
    r"^\d{1,2}-\w{3}-\d{2}",            # page date/header (30-Jan-26...)
    r"^Customer Name,Address",           # column header rows
    r"^Customer Orders",                 # subtitle
    r"^REP_S_\d+",                       # copyright / report footer
    r"^,,Total By Branch",               # branch summary rows
]

BRANCH_NAMES = []  # will be detected dynamically

def classify_line(line: str):
    """Returns 'junk', 'branch_header', or 'data'."""
    stripped = line.strip()
    if not stripped or stripped.replace(",", "").strip() == "":
        return "junk"
    for pat in JUNK_PATTERNS:
        if re.match(pat, stripped):
            return "junk"
    # Branch header: a name followed by only commas/whitespace
    # e.g. "Conut - Tyre,,,,,,,,,"
    # Check on the raw stripped line (before rstrip) so commas are still present
    if re.match(r"^[A-Za-z][A-Za-z0-9 \-]+,{3,}\s*$", stripped):
        return "branch_header"
    return "data"

# ── 3. Parse into structured rows ────────────────────────────────
records = []
current_branch = None

for line in raw_lines:
    kind = classify_line(line)

    if kind == "junk":
        continue

    if kind == "branch_header":
        # Extract branch name (everything before the first comma)
        current_branch = line.strip().split(",")[0].strip()
        if current_branch not in BRANCH_NAMES:
            BRANCH_NAMES.append(current_branch)
        continue

    # ── Parse data row ──
    # Use csv.reader to handle quoted fields like "2,116,800.0"
    parsed = list(csv.reader(StringIO(line.strip())))[0]

    # Strip whitespace from all fields
    parts = [p.strip() for p in parsed]

    # The normal layout is:
    #   0: customer_name, 1: address, 2: phone, 3: first_order, 4: (empty),
    #   5: last_order, 6: (empty), 7: total, 8: num_orders, 9: (empty)
    #
    # Page 15 has an extra empty col, shifting total to idx 8 and num_orders to idx 9:
    #   0: customer_name, 1: address, 2: phone, 3: first_order, 4: (empty),
    #   5: last_order, 6: (empty), 7: (empty), 8: total, 9: num_orders, 10: (empty)

    # Strategy: find the numeric "total" and "num_orders" by scanning from the right
    # num_orders is always last non-empty field (small integer), total is second-to-last
    non_empty = [(i, v) for i, v in enumerate(parts) if v]

    if len(non_empty) < 5:
        print(f"  [SKIPPED] too few fields: {line.strip()[:80]}")
        continue

    customer   = parts[0] if parts[0] else None
    address    = parts[1] if len(parts) > 1 else ""
    phone      = parts[2] if len(parts) > 2 else ""
    first_order = parts[3] if len(parts) > 3 else ""
    # Find last_order: it's at index 5 normally
    last_order = parts[5] if len(parts) > 5 else ""

    # Total and num_orders: grab last two non-empty values
    num_orders_str = non_empty[-1][1]  # last non-empty
    total_str      = non_empty[-2][1]  # second-to-last non-empty

    records.append({
        "branch":      current_branch,
        "customer":    customer,
        # "address":     address,
        # "phone":       phone,
        "first_order": first_order,
        "last_order":  last_order,
        "total":       total_str,
        "num_orders":  num_orders_str,
    })

df = pd.DataFrame(records)
print(f"Parsed rows: {len(df)}")

# ── 4. Clean individual columns ──────────────────────────────────

# Phone: strip whitespace, keep as string
# df["phone"] = df["phone"].str.strip()

# # Address: strip whitespace, replace empty with NaN
# df["address"] = df["address"].str.strip().replace("", pd.NA)

# Timestamps: remove trailing colons, parse to datetime
for col in ["first_order", "last_order"]:
    df[col] = df[col].str.rstrip(":")  # "2025-12-31 19:04:" → "2025-12-31 19:04"
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Total: remove commas from quoted numbers, convert to float
df["total"] = (
    df["total"]
    .str.replace(",", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)

# Num orders: convert to int
df["num_orders"] = pd.to_numeric(df["num_orders"], errors="coerce").astype("Int64")

# ── 5. Drop zero-total rows (cancelled/void orders) ──────────────
zero_mask = df["total"] == 0
print(f"Dropping {zero_mask.sum()} zero-total rows (cancelled/void orders)")
df = df[~zero_mask].copy()

# ── 6. Reset index & summary ─────────────────────────────────────
df = df.reset_index(drop=True)

print(f"\n{'='*55}")
print(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Branches: {df['branch'].nunique()} → {df['branch'].unique().tolist()}")
print(f"Date range: {df['first_order'].min()} → {df['first_order'].max()}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nSample rows:")
print(df.head(8).to_string(index=False))
print(f"\nNull check:\n{df.isnull().sum()}")

# ── 7. Save ──────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved to {OUTPUT_PATH}")
