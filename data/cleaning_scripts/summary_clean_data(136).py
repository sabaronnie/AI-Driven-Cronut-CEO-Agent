"""
Clean & preprocess REP_S_00136_SMRY.csv (Omega POS Summary Report)
-----------------------------------------------------------------
Input:  Messy multi-page POS export with inconsistent columns,
        headers/footers, copyright lines, and page breaks.
Output: Clean CSV with columns:
        [division, category, delivery, takeaway, table, total]
"""

import pandas as pd
import re

INPUT_PATH  = "/Users/ronniesaba/Downloads/Hackathon/Conut bakery Scaled Data /REP_S_00136_SMRY (DONE).csv"  # ← update to your path
OUTPUT_PATH = "/Users/ronniesaba/Downloads/Hackathon/Cleaned_Data/cleaned_revenue_summary(136).csv"                                   # ← update to your path

# ── 1. Read raw lines ────────────────────────────────────────────
with open(INPUT_PATH, "r", encoding="utf-8-sig") as f:
    raw_lines = f.readlines()

print(f"Raw lines read: {len(raw_lines)}")

# ── 2. Filter out junk rows ──────────────────────────────────────
JUNK_PATTERNS = [
    r"^Conut - Tyre,{5,}",
    r"^Summary By Division",
    r"^\d{1,2}-\w{3}-\d{2}",
    r"^,{2,}DELIVERY,TABLE",
    r"^DELIVERY,TABLE",
    r"^REP_S_\d+",
    r"^,{3,}DELIVERY,TABLE",
]

def is_junk(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped == ",":
        return True
    for pat in JUNK_PATTERNS:
        if re.match(pat, stripped):
            return True
    return False

clean_lines = [l for l in raw_lines if not is_junk(l)]
print(f"Lines after junk removal: {len(clean_lines)}")

# ── 3. Parse into structured rows ────────────────────────────────
records = []

for line in clean_lines:
    line = line.strip().rstrip(",").strip()

    parts = []
    in_quote = False
    current = ""
    for ch in line:
        if ch == '"':
            in_quote = not in_quote
        elif ch == "," and not in_quote:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    parts.append(current.strip())

    n = len(parts)

    if n >= 8:
        division  = parts[0]
        category  = parts[1]
        delivery  = parts[3]
        table     = parts[4]
        takeaway  = parts[6]
        total     = parts[7]
    elif n >= 6:
        division  = parts[0]
        category  = parts[1]
        delivery  = parts[2]
        table     = parts[3]
        takeaway  = parts[4]
        total     = parts[5]
    else:
        print(f"  [SKIPPED] ({n} cols): {line[:80]}")
        continue

    records.append({
        "division":  division,
        "category":  category,
        "delivery":  delivery,
        "takeaway":  takeaway,
        "table":     table,
        "total":     total,
    })

df = pd.DataFrame(records)
print(f"Parsed rows: {len(df)}")

# ── 4. Forward-fill division names ────────────────────────────────
df["division"] = df["division"].replace("", pd.NA).ffill()

# ── 5. Clean category names ──────────────────────────────────────
df["category"] = (
    df["category"]
    .str.strip()
    .str.replace("''", "'", regex=False)
)

# ── 6. Convert numeric columns ───────────────────────────────────
num_cols = ["delivery", "takeaway", "table", "total"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── 7. Drop TOTAL summary rows ───────────────────────────────────
total_mask = df["category"].str.upper() == "TOTAL"
print(f"Dropping {total_mask.sum()} TOTAL summary rows")
df = df[~total_mask].copy()

# ── 8. Drop all-zero rows ────────────────────────────────────────
zero_mask = (df[num_cols] == 0).all(axis=1)
print(f"Dropping {zero_mask.sum()} all-zero rows")
df = df[~zero_mask].copy()

# ── 9. Reset index & summary ─────────────────────────────────────
df = df.reset_index(drop=True)

print(f"\n{'='*55}")
print(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Divisions:  {df['division'].nunique()} → {df['division'].unique().tolist()}")
print(f"Categories: {df['category'].nunique()}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nSample rows:")
print(df.head(12).to_string(index=False))
print(f"\nNull check:\n{df.isnull().sum()}")

# ── 10. Save ─────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved to {OUTPUT_PATH}")