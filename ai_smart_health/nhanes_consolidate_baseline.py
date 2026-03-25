from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

BASE_DIR = Path("nhanes_components")
META_DIR = BASE_DIR / "metadata"
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR / "consolidated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Base is the mortality cohort, which has one row per person per cycle, and will be merged with the other files
MORTALITY_COHORT_PATH = Path("nhanes_mortality_data/processed/nhanes_1999_2014_demo_mortality_cohort.csv")

# optional thresholds
MIN_CYCLES_PRESENT = 6
MAX_MISSINGNESS = 0.50
CORR_THRESHOLD = 0.98

ID_COLUMNS = ["SEQN", "cycle"]

LABEL_COLUMNS = [
    "death_1yr",
    "death_3yr",
    "death_5yr",
    "death_10yr",
]

# decide what to remove at that stage
NON_PREDICTOR_COLUMNS = [
    "eligstat",
    #"mortstat",
    "ucod_leading",
    "diabetes_flag",
    "hypertension_flag",
    #"permth_int",
    #"permth_exm",
    "SDMVPSU",
    "SDMVSTRA",
    "WTINT2YR",
    "WTMEC2YR",
]


def read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport")

def file_is_single_row_per_seqn(xpt_path: Path) -> bool:
    try:
        df = read_xpt(xpt_path)
    except Exception:
        return False
    if "SEQN" not in df.columns:
        return False
    return df["SEQN"].nunique(dropna=True) == len(df)

def build_file_level_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Summarizing files"):
        xpt_path_str = row.get("local_xpt_path")
        if not isinstance(xpt_path_str, str) or not xpt_path_str:
            continue

        xpt_path = Path(xpt_path_str)
        try:
            df = read_xpt(xpt_path)
            has_seqn = "SEQN" in df.columns
            n_rows = len(df)
            n_seqn = df["SEQN"].nunique(dropna=True) if has_seqn else np.nan
            one_row_per_seqn = bool(has_seqn and n_rows == n_seqn)
        except Exception:
            has_seqn = False
            n_rows = np.nan
            n_seqn = np.nan
            one_row_per_seqn = False

        rows.append({
            "cycle": row["cycle"],
            "component": row["component"],
            "data_file_code": row["data_file_code"],
            "local_xpt_path": xpt_path_str,
            "n_rows": n_rows,
            "n_seqn": n_seqn,
            "one_row_per_seqn": one_row_per_seqn,
        })
    return pd.DataFrame(rows)

def build_variable_catalog_single_row(manifest: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Building variable catalog"):
        xpt_path_str = row.get("local_xpt_path")
        if not isinstance(xpt_path_str, str) or not xpt_path_str:
            continue

        xpt_path = Path(xpt_path_str)
        try:
            df = read_xpt(xpt_path)
        except Exception:
            continue

        if "SEQN" not in df.columns:
            continue
        if df["SEQN"].nunique(dropna=True) != len(df):
            continue

        for col in df.columns:
            if col == "SEQN":
                continue
            records.append({
                "cycle": row["cycle"],
                "component": row["component"],
                "data_file_code": row["data_file_code"],
                "variable_name": str(col),
                "dtype": str(df[col].dtype),
            })
    return pd.DataFrame(records)

def select_harmonized_variables(variable_catalog: pd.DataFrame, min_cycles_present: int = 6) -> pd.DataFrame:
    presence = (
        variable_catalog.groupby(["variable_name", "component"])["cycle"]
        .nunique()
        .reset_index(name="n_cycles_present")
        .sort_values(["n_cycles_present", "variable_name"], ascending=[False, True])
    )
    keep = presence[presence["n_cycles_present"] >= min_cycles_present].copy()
    return keep

def load_cycle_variables(manifest_cycle: pd.DataFrame, vars_to_keep: set) -> pd.DataFrame:
    """
    Build one harmonized baseline table for a cycle.
    If the same variable occurs in more than one file, keep the first non-missing value across files.
    """
    merged = None

    # stable order
    manifest_cycle = manifest_cycle.sort_values(["component", "data_file_code"]).copy()

    for _, row in tqdm(manifest_cycle.iterrows(), total=len(manifest_cycle), desc="Loading cycle variables"):
        xpt_path_str = row.get("local_xpt_path")
        if not isinstance(xpt_path_str, str) or not xpt_path_str:
            continue

        xpt_path = Path(xpt_path_str)
        try:
            df = read_xpt(xpt_path)
        except Exception:
            continue

        if "SEQN" not in df.columns:
            continue
        if df["SEQN"].nunique(dropna=True) != len(df):
            continue

        cols = ["SEQN"] + [c for c in df.columns if c in vars_to_keep]
        if len(cols) == 1:
            continue

        df = df[cols].copy()

        if merged is None:
            merged = df
        else:
            # combine overlapping variables by first non-missing value
            overlap = [c for c in df.columns if c in merged.columns and c != "SEQN"]
            new_only = [c for c in df.columns if c not in merged.columns]

            merged = merged.merge(df[["SEQN"] + new_only + overlap], on="SEQN", how="outer", suffixes=("", "__new"))

            for c in overlap:
                new_c = f"{c}__new"
                if new_c in merged.columns:
                    merged[c] = merged[c].combine_first(merged[new_c])
                    merged = merged.drop(columns=[new_c])

    if merged is None:
        return pd.DataFrame(columns=["SEQN"])
    return merged

def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if df[c].isna().mean() <= threshold]
    return df[keep_cols].copy()

def drop_highly_correlated_numeric(df: pd.DataFrame, threshold: float = 0.98) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.drop(columns=[c for c in ["SEQN"] if c in numeric.columns], errors="ignore")

    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    return df.drop(columns=to_drop, errors="ignore")

def main():
    manifest = pd.read_csv(META_DIR / "component_file_manifest_downloaded.csv")
    mortality = pd.read_csv(MORTALITY_COHORT_PATH)

    # 1) summarize files
    file_summary = build_file_level_summary(manifest)
    file_summary.to_csv(OUT_DIR / "file_level_summary.csv", index=False)

    # keep only baseline-like files
    manifest_single = manifest.merge(
        file_summary[["cycle", "component", "data_file_code", "one_row_per_seqn"]],
        on=["cycle", "component", "data_file_code"],
        how="left"
    )
    manifest_single = manifest_single[manifest_single["one_row_per_seqn"] == True].copy()

    # 2) variable catalog from baseline-like files only
    variable_catalog = build_variable_catalog_single_row(manifest_single)
    variable_catalog.to_csv(OUT_DIR / "variable_catalog_single_row.csv", index=False)

    # 3) keep variables present in enough cycles
    keep_df = select_harmonized_variables(variable_catalog, min_cycles_present=MIN_CYCLES_PRESENT)
    keep_df.to_csv(OUT_DIR / "variables_present_in_many_cycles.csv", index=False)

    vars_to_keep = set(keep_df["variable_name"].tolist())
    vars_to_keep = {v for v in vars_to_keep if v not in NON_PREDICTOR_COLUMNS}

    # 4) build harmonized baseline table cycle by cycle
    cycle_tables = []
    for cycle in sorted(manifest_single["cycle"].dropna().unique()):
        print(f"[build cycle] {cycle}")
        manifest_cycle = manifest_single[manifest_single["cycle"] == cycle].copy()
        cycle_df = load_cycle_variables(manifest_cycle, vars_to_keep)
        cycle_df["cycle"] = cycle
        cycle_tables.append(cycle_df)

    baseline = pd.concat(cycle_tables, ignore_index=True, sort=False)

    # 5) merge onto mortality cohort
    cohort = mortality.merge(baseline, on=["SEQN", "cycle"], how="left")

    cohort.to_csv(OUT_DIR / "nhanes_baseline_harmonized_before_filters.csv", index=False)

    # 6) predictor set for paper-like reduction
    available_id_cols = [c for c in ID_COLUMNS if c in cohort.columns]
    available_label_cols = [c for c in LABEL_COLUMNS if c in cohort.columns]
    available_non_predictor_cols = [c for c in NON_PREDICTOR_COLUMNS if c in cohort.columns]

    predictor_cols = [
        c for c in cohort.columns
        if c not in available_id_cols + available_label_cols + available_non_predictor_cols
    ]

    id_df = cohort[available_id_cols].copy()
    label_df = cohort[available_label_cols].copy()
    predictors = cohort[predictor_cols].copy()

    predictors_filtered = drop_high_missing_columns(
        predictors,
        threshold=MAX_MISSINGNESS
    )
    predictors_filtered = drop_highly_correlated_numeric(
        predictors_filtered,
        threshold=CORR_THRESHOLD
    )

    predictors_after_filters = pd.concat(
        [id_df, label_df, predictors_filtered],
        axis=1
    )
    predictors_after_filters.to_csv(
        OUT_DIR / "nhanes_predictors_after_missing_corr_filters.csv",
        index=False
    )

    predictors_ohe_only = pd.get_dummies(
        predictors_filtered,
        drop_first=False
    )

    predictors_after_ohe = pd.concat(
        [id_df, label_df, predictors_ohe_only],
        axis=1
    )
    predictors_after_ohe.to_csv(
        OUT_DIR / "nhanes_predictors_after_ohe.csv",
        index=False
    )

    print("Baseline harmonized shape:", baseline.shape)
    print("Cohort shape:", cohort.shape)
    print("Predictors after missing/corr filters:", predictors.shape)
    print("Predictors after one-hot:", predictors_after_ohe.shape)

if __name__ == "__main__":
    main()