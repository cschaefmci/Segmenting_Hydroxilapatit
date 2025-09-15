
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(r"C:\Users\Constantin\Desktop\Segmenting_Hydroxilapatit")
BASE_DIR     = PROJECT_ROOT / r"Test_Results\otsu_threshold_runs"
PRED_SUB     = "thickness_pred"
GT_SUB       = "thickness_gt"
OUT_CSV_NAME = "thickness_metrics_summary.csv"

pred_dir = BASE_DIR / PRED_SUB
gt_dir   = BASE_DIR / GT_SUB

if not pred_dir.is_dir():
    raise NotADirectoryError(f"Vorhersageordner nicht gefunden: {pred_dir}")
if not gt_dir.is_dir():
    raise NotADirectoryError(f"Referenzordner nicht gefunden: {gt_dir}")

#
# Helping Functions
#

def has_header(csv_path: Path) -> bool:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    return bool(re.search(r"[A-Za-zÄÖÜäöü]", first))

def read_thickness_csv(csv_path: Path) -> pd.DataFrame:
    if has_header(csv_path):
        df = pd.read_csv(csv_path)
        cols = list(df.columns)
        if len(cols) < 2:
            df = pd.read_csv(csv_path, header=None, names=["spaltenindex", "dicke_pixel"])
        else:
            mapping = {cols[0]: "spaltenindex", cols[1]: "dicke_pixel"}
            df = df.rename(columns=mapping)[["spaltenindex", "dicke_pixel"]]
    else:
        df = pd.read_csv(csv_path, header=None, names=["spaltenindex", "dicke_pixel"])

    for c in ["spaltenindex", "dicke_pixel"]:
        ser = df[c].astype(str).str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(ser, errors="coerce")

    df = df.drop_duplicates(subset=["spaltenindex"]).sort_values("spaltenindex").reset_index(drop=True)
    return df

def clean_stem(p: Path) -> str:
    stem = p.stem
    stem = re.sub(r"_thr_[0-9]+(?:\.[0-9]+)?$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(?:thickness_)?(pred|gt)$", "", stem, flags=re.IGNORECASE)
    return stem

def compute_metrics(errors: np.ndarray, total_cols: int):
    W = int(np.sum(np.isfinite(errors)))
    if W == 0:
        return dict(W=0, total_columns=int(total_cols), coverage=np.nan,
                    MAE=np.nan, Bias=np.nan, Std=np.nan,
                    CI95_lower=np.nan, CI95_upper=np.nan,
                    MaxAbsError=np.nan)

    e = errors[np.isfinite(errors)]
    mae = float(np.mean(np.abs(e)))
    bias = float(np.mean(e))
    std = float(np.std(e, ddof=1)) if W > 1 else np.nan

    if W > 1 and np.isfinite(std):
        tcrit = stats.t.ppf(0.975, df=W - 1)
        half_width = tcrit * std / math.sqrt(W)
        ci_lo = bias - half_width
        ci_hi = bias + half_width
    else:
        ci_lo = np.nan
        ci_hi = np.nan

    max_abs = float(np.max(np.abs(e)))
    coverage = float(W / total_cols) if total_cols > 0 else np.nan

    return dict(W=W, total_columns=int(total_cols), coverage=coverage,
                MAE=mae, Bias=bias, Std=std,
                CI95_lower=ci_lo, CI95_upper=ci_hi,
                MaxAbsError=max_abs)

# 
# Pair Data
# 

pred_files = sorted(pred_dir.glob("*.csv"))
gt_files   = sorted(gt_dir.glob("*.csv"))

pred_map = {clean_stem(p): p for p in pred_files}
gt_map   = {clean_stem(p): p for p in gt_files}

common_stems = sorted(set(pred_map.keys()) & set(gt_map.keys()))
missing_pred = sorted(set(gt_map.keys()) - set(pred_map.keys()))
missing_gt   = sorted(set(pred_map.keys()) - set(gt_map.keys()))

if missing_pred:
    print("Warnung. Es fehlen Vorhersage Dateien für:", missing_pred)
if missing_gt:
    print("Warnung. Es fehlen Referenz Dateien für:", missing_gt)

# 
# Evaluation
# 

rows = []
all_errors = []

for stem in common_stems:
    df_pred = read_thickness_csv(pred_map[stem]).rename(columns={"dicke_pixel": "d_pred"})
    df_gt   = read_thickness_csv(gt_map[stem]).rename(columns={"dicke_pixel": "d_ref"})

    merged = pd.merge(df_pred[["spaltenindex", "d_pred"]],
                      df_gt[["spaltenindex", "d_ref"]],
                      on="spaltenindex", how="outer", sort=True)

    total_cols = merged["spaltenindex"].nunique()

    valid_mask = merged[["d_pred", "d_ref"]].apply(np.isfinite).all(axis=1)
    merged_valid = merged.loc[valid_mask].copy()
    merged_valid["e"] = merged_valid["d_pred"] - merged_valid["d_ref"]

    m = compute_metrics(merged_valid["e"].to_numpy(), total_cols)
    m["datei_stamm"] = stem
    rows.append(m)

    all_errors.append(merged_valid["e"].to_numpy())

if all_errors:
    all_e = np.concatenate(all_errors)
    total_cols_sum = sum(r["total_columns"] for r in rows)
    overall = compute_metrics(all_e, total_cols_sum)
    overall["datei_stamm"] = "ALLE_DATEIEN"
    rows.append(overall)

# 
# Create Table
# 

cols_order = ["datei_stamm", "W", "total_columns", "coverage",
              "MAE", "Bias", "Std", "CI95_lower", "CI95_upper", "MaxAbsError"]

result_df = pd.DataFrame.from_records(rows)

if result_df.empty:
    print("Keine Paare ausgewertet. Prüfe Dateinamen in thickness_pred und thickness_gt.")
    result_df = pd.DataFrame(columns=cols_order)
else:
    for c in cols_order:
        if c not in result_df:
            result_df[c] = np.nan
    result_df = result_df[cols_order]

for c in ["coverage", "MAE", "Bias", "Std", "CI95_lower", "CI95_upper", "MaxAbsError"]:
    if c in result_df.columns:
        result_df[c] = result_df[c].astype(float).round(6)

out_path = BASE_DIR / OUT_CSV_NAME
result_df.to_csv(out_path, index=False, encoding="utf-8")
print(f"Fertig. Zusammenfassung gespeichert unter:\n{out_path}")

with pd.option_context("display.width", None, "display.max_columns", None):
    print(result_df.head(10))