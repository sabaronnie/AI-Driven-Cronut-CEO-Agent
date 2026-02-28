"""
Clean & preprocess rep_s_00191_SMRY.csv (Omega POS Sales by Items By Group)
---------------------------------------------------------------------------
Input:  Messy multi-page POS export with Branch/Division/Group hierarchy,
        repeated page headers, totals rows, and copyright footer.
Output: Clean CSV with columns:
        [division, category_name, category_group, channel, revenue]
"""

import pandas as pd
import re
import csv
from io import StringIO

INPUT_PATH  = "/Users/ronniesaba/Downloads/Hackathon/Conut bakery Scaled Data /rep_s_00191_SMRY.csv"   # ← update to your path
OUTPUT_PATH = "/Users/ronniesaba/Downloads/Hackathon/Cleaned_Data/cleaned_sales_by_item.csv"                       # ← update to your path

# ── 1. Read raw lines ────────────────────────────────────────────
with open(INPUT_PATH, "r", encoding="utf-8-sig") as f:
    raw_lines = f.readlines()

print(f"Raw lines read: {len(raw_lines)}")

# ── 2. Junk patterns ─────────────────────────────────────────────
JUNK_PATTERNS = [
    r"^\d{1,2}-\w{3}-\d{2}",        # page date header (30-Jan-26...)
    r"^Description,Barcode",          # column header rows
    r"^Sales by Items",               # subtitle
    r"^REP_S_\d+",                    # copyright / report footer
    r"^Total by ",                    # all aggregate rows (Group/Division/Branch)
    r"^,{2,}www\.",                   # trailing URL line
]

def is_junk(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.replace(",", "").strip() == "":
        return True
    for pat in JUNK_PATTERNS:
        if re.match(pat, stripped):
            return True
    return False

# ── 3. Parse into structured rows ────────────────────────────────
records = []
current_channel = None
current_division = None
current_group = None

# Also match the title row (e.g. "Conut - Tyre,,,,")
TITLE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9 \-'']+,{3,}\s*$")

for line in raw_lines:
    stripped = line.strip()

    # Skip junk
    if is_junk(stripped):
        continue

    # Title row (report title, same as first branch name — skip)
    if TITLE_PATTERN.match(stripped):
        continue

    # Branch header: "Branch: Conut - Tyre,,,,"
    if stripped.startswith("Branch:"):
        current_channel = stripped.split(":", 1)[1].split(",")[0].strip()
        continue

    # Division header: "Division: Hot-Coffee Based,,,,"
    if stripped.startswith("Division:"):
        current_division = stripped.split(":", 1)[1].split(",")[0].strip()
        continue

    # Group header: "Group: Hot-Coffee Based,,,,"
    if stripped.startswith("Group:"):
        current_group = stripped.split(":", 1)[1].split(",")[0].strip()
        continue

    # ── Data row ──
    parsed = list(csv.reader(StringIO(stripped)))[0]
    parts = [p.strip() for p in parsed]

    if len(parts) < 4:
        print(f"  [SKIPPED] too few fields: {stripped[:80]}")
        continue

    category_name = parts[0]
    # parts[1] = barcode (always empty), parts[2] = qty
    revenue_str = parts[3] if len(parts) > 3 else "0"

    if not category_name:
        continue

    records.append({
        "division":       current_division,
        "category_name":  category_name,
        "channel":        current_channel,
        "revenue":        revenue_str,
    })

df = pd.DataFrame(records)
print(f"Parsed rows: {len(df)}")

# ── 4. Clean columns ─────────────────────────────────────────────

# Fix double single-quotes (CONUT''S → CONUT'S)
for col in ["division", "category_name"]:
    df[col] = df[col].str.replace("''", "'", regex=False)

# Revenue: remove commas, convert to float
df["revenue"] = (
    df["revenue"]
    .str.replace(",", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)

# ── 5. Drop zero-revenue rows (free options / no sales) ──────────
zero_mask = df["revenue"] == 0
print(f"Dropping {zero_mask.sum()} zero-revenue rows")
df = df[~zero_mask].copy()

# ── 6. Reset index & summary ─────────────────────────────────────
df = df.reset_index(drop=True)

print(f"\n{'='*55}")
print(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Channels:   {df['channel'].nunique()} → {df['channel'].unique().tolist()}")
print(f"Divisions:  {df['division'].nunique()}")
print(f"Items:      {df['category_name'].nunique()}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nSample rows:")
print(df.head(10).to_string(index=False))
print(f"\nNull check:\n{df.isnull().sum()}")

# ── 7. Save ──────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved to {OUTPUT_PATH}")