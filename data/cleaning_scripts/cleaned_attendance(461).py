"""
Clean & preprocess REP_S_00461.csv (Omega POS Time & Attendance Report)
-----------------------------------------------------------------------
Input:  Multi-page employee punch-in/out report with employee headers,
        branch assignments, page headers, totals, and copyright footer.
Output: Clean CSV with columns:
        [branch, date_in, punch_in, date_out, punch_out, work_duration]
"""

import pandas as pd
import re

INPUT_PATH  = "/Users/ronniesaba/Downloads/Hackathon/Conut bakery Scaled Data /REP_S_00461.csv"  # ← update to your path
OUTPUT_PATH = "/Users/ronniesaba/Downloads/Hackathon/Cleaned_Data/cleaned_attendance(461).csv"                    # ← update to your path

# ── 1. Read raw lines ────────────────────────────────────────────
with open(INPUT_PATH, "r", encoding="utf-8-sig") as f:
    raw_lines = f.readlines()

print(f"Raw lines read: {len(raw_lines)}")

# ── 2. Junk patterns ─────────────────────────────────────────────
JUNK_PATTERNS = [
    r"^Conut - Tyre,",                  # title row
    r"^Time & Attendance",               # subtitle
    r"^,30-Jan-26",                      # page header
    r"^,PUNCH IN",                       # column header
    r"^REP_S_\d+",                       # copyright
    r"^,EMP ID",                         # employee header
    r"^,,,,Total",                       # employee total
]

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

# Branch line pattern: ",Conut Jnah,,,," or ",Main Street Coffee,,,,"
BRANCH_PATTERN = re.compile(r"^,([A-Za-z][A-Za-z0-9 \-]+),{3,}")

for line in raw_lines:
    stripped = line.strip()

    if is_junk(stripped):
        continue

    # Branch assignment line: ",Conut Jnah,,,,"
    m = BRANCH_PATTERN.match(stripped)
    if m:
        current_branch = m.group(1).strip()
        continue

    # Data row: 01-Dec-25,,07.39.35,01-Dec-25,19.37.56,11.58.21
    parts = [p.strip() for p in stripped.split(",")]

    if len(parts) >= 6 and re.match(r"^\d{2}-\w{3}-\d{2}$", parts[0]):
        records.append({
            "branch":        current_branch,
            "date_in":       parts[0],
            "punch_in":      parts[2],
            "date_out":      parts[3],
            "punch_out":     parts[4],
            "work_duration": parts[5],
        })

df = pd.DataFrame(records)
print(f"Parsed rows: {len(df)}")

# ── 4. Clean columns ─────────────────────────────────────────────

# Convert dates
for col in ["date_in", "date_out"]:
    df[col] = pd.to_datetime(df[col], format="%d-%b-%y", errors="coerce")

# Convert time strings from HH.MM.SS to HH:MM:SS
for col in ["punch_in", "punch_out", "work_duration"]:
    df[col] = df[col].str.replace(".", ":", regex=False)

# ── 5. Summary ───────────────────────────────────────────────────
df = df.reset_index(drop=True)

print(f"\n{'='*55}")
print(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Branches: {df['branch'].nunique()} → {df['branch'].unique().tolist()}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nSample rows:")
print(df.head(10).to_string(index=False))
print(f"\nNull check:\n{df.isnull().sum()}")

# ── 6. Save ──────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved to {OUTPUT_PATH}")