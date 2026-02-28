"""
Clean & preprocess REP_S_00502.csv (Omega POS Sales by Customer in Details - Delivery)
---------------------------------------------------------------------------------------
Input:  Messy multi-page POS export with Branch/Person hierarchy, per-item lines,
        per-person totals, page headers, and copyright footer.
Output: Clean CSV with columns:
        [branch, person, from_date, to_date, description, qty, price, total]
        where 'total' is the person's order total attached to each line item.
"""

import pandas as pd
import re
import csv
from io import StringIO

INPUT_PATH  = "/Users/ronniesaba/Downloads/Hackathon/Conut bakery Scaled Data /REP_S_00502.csv"   # ← update to your path
OUTPUT_PATH = "/Users/ronniesaba/Downloads/Hackathon/Cleaned_Data/cleaned_sales_by_customer(502).csv"              # ← update to your path

# ── 1. Read raw lines ────────────────────────────────────────────
with open(INPUT_PATH, "r", encoding="utf-8-sig") as f:
    raw_lines = f.readlines()

print(f"Raw lines read: {len(raw_lines)}")

# ── 2. Extract global from_date / to_date from page header ───────
from_date = None
to_date = None
for line in raw_lines:
    m = re.search(r"From Date:\s*([\d\-A-Za-z]+)", line)
    if m:
        from_date = m.group(1)
    m2 = re.search(r"To Date:\s*([\d\-A-Za-z]+)", line)
    if m2:
        to_date = m2.group(1)
    if from_date and to_date:
        break

# Parse to proper dates
from_date = pd.to_datetime(from_date, format="%d-%b-%Y", errors="coerce")
to_date   = pd.to_datetime(to_date, format="%d-%b-%Y", errors="coerce")
print(f"Date range: {from_date.date()} → {to_date.date()}")

# ── 3. Junk detection ────────────────────────────────────────────
JUNK_PATTERNS = [
    r"^\d{1,2}-\w{3}-\d{2}",        # page date header
    r"^Full Name,Qty",               # column header
    r"^Sales by customer",           # subtitle
    r"^REP_S_\d+",                   # copyright footer
    r"^Total Branch:",               # branch-level aggregate
]

def is_junk(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.replace(",", "").strip() == "":
        return True
    for pat in JUNK_PATTERNS:
        if re.match(pat, stripped):
            return True
    return False

# ── 4. Parse ─────────────────────────────────────────────────────
# Strategy: buffer items per person, flush when we hit "Total :" row
records = []
current_branch = None
current_person = None
person_items = []  # buffered items for current person

# Also match title row (e.g. "Conut - Tyre,,,,")
TITLE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9 \-]+,{3,}\s*$")

def flush_person(items, person_total):
    """Attach person total to each buffered item and add to records."""
    for item in items:
        item["total"] = person_total
        records.append(item)

for line in raw_lines:
    stripped = line.strip()

    if is_junk(stripped):
        continue

    if TITLE_PATTERN.match(stripped):
        continue

    # Branch header: "Branch :Conut - Tyre,,,,"
    if stripped.startswith("Branch"):
        current_branch = stripped.split(":", 1)[1].split(",")[0].strip()
        continue

    # Person header: "Person_0129,,,,"
    if re.match(r"^Person_\d+,", stripped):
        current_person = stripped.split(",")[0].strip()
        person_items = []
        continue

    # Parse with csv.reader for quoted fields
    parsed = list(csv.reader(StringIO(stripped)))[0]
    parts = [p.strip() for p in parsed]

    # Total row: "Total :,10.0,,2443378.38,"
    if parts[0].startswith("Total"):
        total_str = parts[3] if len(parts) > 3 else "0"
        total_val = float(total_str.replace(",", "")) if total_str else 0.0
        flush_person(person_items, total_val)
        person_items = []
        continue

    # Item row: ,qty,description,price,
    if len(parts) >= 4:
        qty_str = parts[1]
        description = parts[2]
        price_str = parts[3]

        person_items.append({
            "branch":      current_branch,
            "person":      current_person,
            "from_date":   from_date,
            "to_date":     to_date,
            "description": description,
            "qty":         qty_str,
            "price":       price_str,
        })

df = pd.DataFrame(records)
print(f"Parsed rows: {len(df)}")

# ── 5. Clean columns ─────────────────────────────────────────────

# Description: strip leading whitespace and brackets
df["description"] = df["description"].str.strip()

# Qty: convert to float
df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

# Price: remove commas, convert to float
df["price"] = (
    df["price"]
    .str.replace(",", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)

# ── 6. Drop fully empty item rows (malformed data) ───────────────
na_mask = df["description"].isna() | (df["description"] == "") | df["qty"].isna()
print(f"Dropping {na_mask.sum()} empty/malformed item rows")
df = df[~na_mask].copy()

# ── 7. Drop zero-price rows (free options/toppings) ──────────────
zero_mask = df["price"] == 0
print(f"Dropping {zero_mask.sum()} zero-price rows (free options)")
df = df[~zero_mask].copy()

# ── 8. Reset index & summary ─────────────────────────────────────
df = df.reset_index(drop=True)

print(f"\n{'='*55}")
print(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Branches: {df['branch'].nunique()} → {df['branch'].unique().tolist()}")
print(f"Persons:  {df['person'].nunique()}")
print(f"Items:    {df['description'].nunique()}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nSample rows:")
print(df.head(10).to_string(index=False))
print(f"\nNull check:\n{df.isnull().sum()}")

# ── 9. Save ──────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved to {OUTPUT_PATH}")