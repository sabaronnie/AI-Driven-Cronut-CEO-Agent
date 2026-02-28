"""
Clean & preprocess rep_s_00435_SMRY.csv (Omega POS Average Sales By Menu)
-------------------------------------------------------------------------
Input:  Small multi-branch POS summary with branch headers, totals, and junk.
Output: Clean CSV with columns:
        [branch, channel, num_customers, sales, avg_customer, total_customers, total_sales, total_avg_customer]
"""

import pandas as pd
import re
import csv
from io import StringIO

INPUT_PATH  = "/Users/ronniesaba/Downloads/Hackathon/Conut bakery Scaled Data /rep_s_00435_SMRY.csv"  # ← update to your path
OUTPUT_PATH = "/Users/ronniesaba/Downloads/Hackathon/Cleaned_Data/cleaned_avg_sales_by_menu(435).csv"                  # ← update to your path

# ── 1. Read raw lines ────────────────────────────────────────────
with open(INPUT_PATH, "r", encoding="utf-8-sig") as f:
    raw_lines = f.readlines()

print(f"Raw lines read: {len(raw_lines)}")

# ── 2. Junk patterns ─────────────────────────────────────────────
JUNK_PATTERNS = [
    r"^\d{1,2}-\w{3}-\d{2}",        # date row
    r"^Menu Name,",                   # column header
    r"^Average Sales",                # subtitle
    r"^REP_S_\d+",                    # copyright
    r"^Total :",                      # grand total
    r"^,Year:",                       # year/page header
]

TITLE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9 \-]+,{3,}\s*$")

def is_junk(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.replace(",", "").strip() == "":
        return True
    for pat in JUNK_PATTERNS:
        if re.match(pat, stripped):
            return True
    return False

# ── 3. Parse ─────────────────────────────────────────────────────
records = []
current_branch = None
branch_items = []  # buffer items per branch

for line in raw_lines:
    stripped = line.strip()

    if is_junk(stripped):
        continue

    # Branch header: "Conut - Tyre,,,,"
    if TITLE_PATTERN.match(stripped):
        current_branch = stripped.split(",")[0].strip()
        branch_items = []
        continue

    # Parse row
    parsed = list(csv.reader(StringIO(stripped)))[0]
    parts = [p.strip() for p in parsed]

    if len(parts) < 4:
        print(f"  [SKIPPED]: {stripped[:80]}")
        continue

    # Total By Branch row — flush with these totals
    if parts[0].startswith("Total By Branch"):
        total_cust_str = parts[1].replace(",", "")
        total_sales_str = parts[2].replace(",", "")
        total_avg_str = parts[3].replace(",", "")
        t_cust = float(total_cust_str) if total_cust_str else 0.0
        t_sales = float(total_sales_str) if total_sales_str else 0.0
        t_avg = float(total_avg_str) if total_avg_str else 0.0
        for item in branch_items:
            item["total_customers"] = t_cust
            item["total_sales"] = t_sales
            item["total_avg_customer"] = t_avg
            records.append(item)
        branch_items = []
        continue

    branch_items.append({
        "branch":        current_branch,
        "channel":       parts[0],
        "num_customers": parts[1],
        "sales":         parts[2],
        "avg_customer":  parts[3],
    })

df = pd.DataFrame(records)
print(f"Parsed rows: {len(df)}")

# ── 4. Clean columns ─────────────────────────────────────────────
for col in ["num_customers", "sales", "avg_customer"]:
    df[col] = df[col].str.replace(",", "", regex=False).pipe(pd.to_numeric, errors="coerce")
# total_customers, total_sales, total_avg_customer are already numeric from parsing

# ── 5. Ensure all branches have all 3 channels ───────────────────
all_channels = ["DELIVERY", "TABLE", "TAKE AWAY"]
all_branches = df["branch"].unique()

full_index = pd.MultiIndex.from_product([all_branches, all_channels], names=["branch", "channel"])
df = df.set_index(["branch", "channel"]).reindex(full_index, fill_value=0).reset_index()

# ── 6. Summary ───────────────────────────────────────────────────
df = df.reset_index(drop=True)

print(f"\n{'='*55}")
print(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Branches: {df['branch'].nunique()} → {df['branch'].unique().tolist()}")
print(f"Channels: {df['channel'].unique().tolist()}")
print(f"\n{df.to_string(index=False)}")
print(f"\nNull check:\n{df.isnull().sum()}")

# ── 7. Save ──────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved to {OUTPUT_PATH}")
