"""
Clean & preprocess REP_S_00194_SMRY.csv (Omega POS Tax Report)
--------------------------------------------------------------
Input:  Tax report with branch-level VAT totals.
Output: Clean CSV with columns:
        [branch, vat_11_total]
"""

import pandas as pd
import csv
from io import StringIO

INPUT_PATH  = "/Users/ronniesaba/Downloads/Hackathon/Conut bakery Scaled Data /REP_S_00194_SMRY.csv"  # ← update to your path
OUTPUT_PATH = "/Users/ronniesaba/Downloads/Hackathon/Cleaned_Data/cleaned_tax_report.csv"                         # ← update to your path

# ── 1. Read raw lines ────────────────────────────────────────────
with open(INPUT_PATH, "r", encoding="utf-8-sig") as f:
    raw_lines = f.readlines()

print(f"Raw lines read: {len(raw_lines)}")

# ── 2. Parse ─────────────────────────────────────────────────────
records = []
current_branch = None

for line in raw_lines:
    stripped = line.strip()

    # Branch header: "Branch Name:  Conut,,,,,,,,,
    if stripped.startswith("Branch Name:"):
        current_branch = stripped.split(":", 1)[1].split(",")[0].strip()
        continue

    # Data row: "Total By Branch,"427,048,534.35",..."
    if stripped.startswith("Total By Branch"):
        parsed = list(csv.reader(StringIO(stripped)))[0]
        vat_str = parsed[1].strip() if len(parsed) > 1 else "0"
        records.append({
            "branch":      current_branch,
            "vat_11_total": vat_str,
        })

df = pd.DataFrame(records)

# ── 3. Clean ─────────────────────────────────────────────────────
df["vat_11_total"] = df["vat_11_total"].str.replace(",", "", regex=False).pipe(pd.to_numeric, errors="coerce")

# ── 4. Summary ───────────────────────────────────────────────────
print(f"\n{df.to_string(index=False)}")
print(f"\nNull check:\n{df.isnull().sum()}")

# ── 5. Save ──────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved to {OUTPUT_PATH}")