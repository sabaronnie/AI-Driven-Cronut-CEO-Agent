"""
Clean & preprocess rep_s_00334_1_SMRY.csv (Omega POS Monthly Sales)
-------------------------------------------------------------------
Input:  Multi-branch monthly sales with branch headers, yearly/branch totals,
        page headers, and copyright footer.
Output: Clean CSV with columns:
        [branch, month, year, sales, total]
"""

import pandas as pd
import re
import csv
from io import StringIO

INPUT_PATH  = "/Users/ronniesaba/Downloads/Hackathon/Conut bakery Scaled Data /rep_s_00334_1_SMRY.csv"  # ← update to your path
OUTPUT_PATH = "/Users/ronniesaba/Downloads/Hackathon/Cleaned_Data/cleaned_monthly_sales(334).csv"                        # ← update to your path

# ── 1. Read raw lines ────────────────────────────────────────────
with open(INPUT_PATH, "r", encoding="utf-8-sig") as f:
    raw_lines = f.readlines()

print(f"Raw lines read: {len(raw_lines)}")

# ── 2. Junk patterns ─────────────────────────────────────────────
JUNK_PATTERNS = [
    r"^\d{1,2}-\w{3}-\d{2}",        # date row
    r"^Month,,Year",                  # column header
    r"^Monthly Sales",                # subtitle
    r"^REP_S_\d+",                    # copyright
    r"^,Year:",                       # page subheader
    r"^,,Total for",                  # yearly total
    r"^,,Grand Total",                # grand total
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
branch_items = []

for line in raw_lines:
    stripped = line.strip()

    if is_junk(stripped):
        continue

    # Title row (report name)
    if TITLE_PATTERN.match(stripped):
        continue

    # Branch header: "Branch Name: Conut,,,,"
    if stripped.startswith("Branch Name:"):
        current_branch = stripped.split(":", 1)[1].split(",")[0].strip()
        branch_items = []
        continue

    parsed = list(csv.reader(StringIO(stripped)))[0]
    parts = [p.strip() for p in parsed]

    # Total by Branch row — flush with this total
    if len(parts) >= 4 and parts[2].startswith("Total by Branch"):
        total_str = parts[3].replace(",", "")
        branch_total = float(total_str) if total_str else 0.0
        for item in branch_items:
            item["total"] = branch_total
            records.append(item)
        branch_items = []
        continue

    # Data row: August,,2025,"554,074,782.88",
    if len(parts) >= 4 and parts[0]:
        branch_items.append({
            "branch": current_branch,
            "month":  parts[0],
            "year":   parts[2],
            "sales":  parts[3],
        })

df = pd.DataFrame(records)
print(f"Parsed rows: {len(df)}")

# ── 4. Clean columns ─────────────────────────────────────────────
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

for col in ["sales"]:
    df[col] = df[col].str.replace(",", "", regex=False).pipe(pd.to_numeric, errors="coerce")

# total is already numeric from parsing

# ── 5. Fill missing months with 0 ────────────────────────────────
all_months = ["August", "September", "October", "November", "December"]
all_branches = df["branch"].unique()

full_index = pd.MultiIndex.from_product([all_branches, all_months], names=["branch", "month"])
df = df.set_index(["branch", "month"]).reindex(full_index, fill_value=0).reset_index()

# Re-fill year for added rows
df["year"] = df["year"].replace(0, 2025)

# ── 6. Summary ───────────────────────────────────────────────────
df = df.reset_index(drop=True)

print(f"\n{'='*55}")
print(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Branches: {df['branch'].nunique()} → {df['branch'].unique().tolist()}")
print(f"Months:   {df['month'].unique().tolist()}")
print(f"\n{df.to_string(index=False)}")
print(f"\nNull check:\n{df.isnull().sum()}")

# ── 7. Save ──────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved to {OUTPUT_PATH}")