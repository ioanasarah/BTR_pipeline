"""
consensus_mz_features.py
------------------------
Finds m/z values that appear as top RF features in more than 75% of
pra (pre-perfusion) files OR more than 75% of 1hnr (post-perfusion) files.

Expects CSV files named like:
    rf_top_features_DHB_1_pra_OMP_spca10_k4.csv
    rf_top_features_DHB_21_1hnr_OMP_spca10_k4.csv

Expected CSV columns (adjust MZ_COL / IMPORTANCE_COL if different):
    mz, importance
"""
import os
import glob
import pandas as pd
from collections import defaultdict
from pathlib import Path

from sympy import re


# print(os.path.exists(r"C:\Ioana\GitHub\BTR_pipeline\results\liver_pc\OMP_spca10_k4"))




# files = glob.glob(r"C:\Ioana\GitHub\BTR_pipeline\results\liver_pc\OMP_spca10_k4\**\*.csv", recursive=True)
# print(len(files))
# for f in files:
#     print(f)



# ── CONFIG ─────────────────────────────────────────────────────────────────────
# RESULTS_FOLDER = r"C:\Ioana\GitHub\BTR_pipeline\results\liver_pc\OMP_spca10_k4"  # change as needed
RESULTS_FOLDER = r"C:\Users\i6338212\data\results\liver_PC\OMP_pca10_kmeans4_smoothing"
CSV_PATTERN = "**/*rf_feature*.csv"  # recursive search
THRESHOLD      = 0.4                      # 75%
MZ_COL         = "mz"                          # column name for m/z values
IMPORTANCE_COL = "importance"                  # column name for importance score
MZ_TOLERANCE   = 0.01                          # Da — set to 0 for exact string match

# Keywords used to identify the condition from the filename
PRA_KEYWORD  = "pra"
HNRR_KEYWORD = "1hnr"
# ───────────────────────────────────────────────────────────────────────────────



files = list(Path(RESULTS_FOLDER).rglob("**/*rf_top_features*.csv"))
print(len(files))

def find_condition(filename: str) -> str | None:
    """Return 'pra', '1hnr', or None if neither keyword found in filename."""
    name = os.path.basename(filename).lower()
    if PRA_KEYWORD in name:
        return "pra"
    if HNRR_KEYWORD in name:
        return "1hnr"
    return None


def load_mz_values(csv_path: str) -> list[float]:
    """Load m/z values from a single RF results CSV."""
    df = pd.read_csv(csv_path)
    if MZ_COL not in df.columns:
        raise ValueError(f"Column '{MZ_COL}' not found in {csv_path}.\n"
                         f"Available columns: {list(df.columns)}")
    return df[MZ_COL].dropna().tolist()


def group_files_by_condition(csv_files: list[str]) -> dict[str, list[str]]:
    """Split file list into pra and 1hnr groups."""
    groups = defaultdict(list)
    skipped = []
    for f in csv_files:
        condition = find_condition(f)
        if condition:
            groups[condition].append(f)
        else:
            skipped.append(f)
    if skipped:
        print(f"  Skipped {len(skipped)} files (condition not detected in name):")
        for s in skipped:
            print(f"    {os.path.basename(s)}")
    return dict(groups)


def match_mz(target: float, reference_list: list[float], tol: float) -> float | None:
    """Return the closest m/z in reference_list within tolerance, or None."""
    if tol == 0:
        return target if target in reference_list else None
    candidates = [r for r in reference_list if abs(r - target) <= tol]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs(r - target))


def find_consensus_mz(
    files: list[str],
    threshold: float,
    tol: float
) -> pd.DataFrame:
    """
    For a group of CSV files, find m/z values that appear in more than
    `threshold` fraction of files.

    Returns a DataFrame with columns:
        mz, n_files_present, pct_files_present, mean_importance, files_present
    """
    if not files:
        return pd.DataFrame()

    # Load all files
    all_mz_lists = []
    file_labels  = []
    importance_lookup = {}   # mz (rounded) -> list of importance values

    for f in files:
        df = pd.read_csv(f)

        # Keep only top 20 features based on rank
        if "rank" in df.columns:
            df = df[df["rank"] <= 20]
        else:
            # fallback: just take first 20 rows if already sorted
            df = df.head(20)

        if MZ_COL not in df.columns:
            print(f"  WARNING: '{MZ_COL}' not found in {os.path.basename(f)}, skipping.")
            continue
        mz_vals = df[MZ_COL].dropna().tolist()
        all_mz_lists.append(mz_vals)
        file_labels.append(os.path.basename(f))

        # Store importance if available
        if IMPORTANCE_COL in df.columns:
            for _, row in df.iterrows():
                key = round(float(row[MZ_COL]), 4)
                importance_lookup.setdefault(key, []).append(float(row[IMPORTANCE_COL]))

    n_files = len(all_mz_lists)
    min_count = threshold * n_files

    # Build a unified m/z reference pool from the first file,
    # then count appearances across all files using tolerance matching
    # (if all files share the same peak list, every value will match exactly)
    reference_pool = list(set(
        round(mz, 4) for mz_list in all_mz_lists for mz in mz_list
    ))

    mz_counts    = defaultdict(int)
    mz_files     = defaultdict(list)

    for mz_list, fname in zip(all_mz_lists, file_labels):
        seen_in_this_file = set()
        for mz in mz_list:
            matched = match_mz(round(mz, 4), reference_pool, tol)
            if matched is not None and matched not in seen_in_this_file:
                mz_counts[matched] += 1
                mz_files[matched].append(fname)
                seen_in_this_file.add(matched)

    # Filter by threshold
    rows = []
    for mz, count in mz_counts.items():
        if count >= min_count:
            imp_vals = importance_lookup.get(mz, [])
            rows.append({
                "mz":                 mz,
                "n_files_present":    count,
                "pct_files_present":  round(count / n_files * 100, 1),
                "mean_importance":    round(sum(imp_vals) / len(imp_vals), 6) if imp_vals else None,
                "files_present":      "; ".join(mz_files[mz])
            })

    if not rows:
        return pd.DataFrame(columns=[
        "mz", "n_files_present", "pct_files_present",
        "mean_importance", "files_present"
    ])

    df_out = pd.DataFrame(rows).sort_values("pct_files_present", ascending=False)
    return df_out.reset_index(drop=True)


def run(results_folder: str,
        csv_pattern: str,
        threshold: float,
        tol: float) -> None:

    # Find all RF CSVs
    search_path = os.path.join(results_folder, csv_pattern)
    print("Search path:", search_path)
    csv_files   = glob.glob(search_path, recursive=True)
    print(f"Found {len(csv_files)} RF results CSVs under {results_folder}")

    if not csv_files:
        print("No files found. Check RESULTS_FOLDER and CSV_PATTERN.")
        return

    # Group by condition
    groups = group_files_by_condition(csv_files)
    print(f"\nCondition breakdown:")
    for cond, flist in groups.items():
        print(f"  {cond}: {len(flist)} files")
        for f in flist:
            print(f"    {os.path.basename(f)}")

    # Find consensus for each condition
    summary = {}
    for condition, flist in groups.items():
        print(f"\n{'─'*60}")
        print(f"Processing condition: {condition.upper()}  ({len(flist)} files)")
        print(f"Threshold: >{threshold*100:.0f}% = must appear in >{threshold*len(flist):.1f} files")

        df_consensus = find_consensus_mz(flist, threshold, tol)
        summary[condition] = df_consensus

        print(f"\nConsensus m/z values ({len(df_consensus)} found):")
        if df_consensus.empty:
            print("  None found above threshold.")
        else:
            print(df_consensus[["mz", "n_files_present", "pct_files_present", "mean_importance"]].to_string(index=False))

        # Save per-condition CSV
        out_path = os.path.join(results_folder, f"consensus_mz_{condition}.csv")
        df_consensus.to_csv(out_path, index=False)
        print(f"\nSaved to: {out_path}")

    # Save combined comparison if both conditions present
    if "pra" in summary and "1hnr" in summary:
        pra_mz  = set(summary["pra"]["mz"])
        hnr_mz  = set(summary["1hnr"]["mz"])
        shared  = pra_mz & hnr_mz
        pra_only  = pra_mz - hnr_mz
        hnr_only  = hnr_mz - pra_mz

        print(f"\n{'═'*60}")
        print(f"CROSS-CONDITION SUMMARY")
        print(f"  Shared in both pra AND 1hnr consensus: {len(shared)}")
        print(f"  Unique to pra only:                    {len(pra_only)}")
        print(f"  Unique to 1hnr only:                   {len(hnr_only)}")

        if shared:
            print(f"\n  Shared m/z values: {sorted(shared)}")

        # Save shared
        shared_path = os.path.join(results_folder, "consensus_mz_shared_pra_1hnr.csv")
        pd.DataFrame({"mz": sorted(shared)}).to_csv(shared_path, index=False)
        print(f"\nShared m/z saved to: {shared_path}")


if __name__ == "__main__":
    run(
        results_folder=RESULTS_FOLDER,
        csv_pattern=CSV_PATTERN,
        threshold=THRESHOLD,
        tol=MZ_TOLERANCE
    )
