from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
from tqdm import tqdm

BASE_DIR = Path("nhanes_components")
RAW_DIR = BASE_DIR / "raw"
META_DIR = BASE_DIR / "metadata"
OUT_DIR = BASE_DIR / "lookup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns of interest (after consolidation step)
TARGET_VARS = [
    'age', 'sex', 'race_ethnicity', 'education', 'marital_status',
    'poverty_income_ratio', 'weight_interview_2yr', 'weight_exam_2yr',
    'death_10yr', 'household_income_cat', 'SDDSRVYR', 'RIDSTATR',
    'RIDEXMON', 'DMQMILIT', 'DMDCITZN', 'DMDHHSIZ', 'DMDHRGND',
    'DMDHRAGE', 'DMDHREDU', 'DMDHRMAR', 'BMXWT', 'BMXHT', 'BMXBMI',
    'BMXLEG', 'BMXARML', 'BMXARMC', 'BMXWAIST', 'BMXTRI', 'BMXSUB',
    'PEASCST1', 'PEASCTM1', 'BPQ150A', 'BPQ150B', 'BPQ150C', 'BPQ150D',
    'BPAARM', 'BPACSZ', 'BPXPLS', 'BPXPULS', 'BPXPTY', 'BPXML1',
    'BPXSY1', 'BPXDI1', 'BPAEN1', 'BPXSY2', 'BPXDI2', 'BPAEN2',
    'BPXSY3', 'BPXDI3', 'BPAEN3', 'OHXIMP', 'OHAREC', 'LBXHBS',
    'LBXHA', 'LBXHBC', 'LBDHBG', 'LBDHCV', 'LBDHD', 'LBDHI', 'LBXBPB',
    'LBXBCD', 'LBDB12SI', 'LBXTHG', 'LBXIHG', 'LBXCOT', 'URXUCR',
    'LBXGH', 'LBXCRP', 'LBXTC', 'URXUMA', 'LBXSAL', 'LBXSATSI',
    'LBXSASSI', 'LBXSAPSI', 'LBXSBU', 'LBXSCA', 'LBXSC3SI', 'LBXSGTSI',
    'LBXSGL', 'LBXSIR', 'LBXSLDSI', 'LBXSPH', 'LBXSTB', 'LBXSTP',
    'LBXSTR', 'LBXSUA', 'LBXSCR', 'LBXSNASI', 'LBXSKSI', 'LBXSCLSI',
    'LBXSOSSI', 'LBXSGB', 'LBXWBCSI', 'LBXLYPCT', 'LBXMOPCT',
    'LBXNEPCT', 'LBXEOPCT', 'LBXBAPCT', 'LBDLYMNO', 'LBDMONO',
    'LBDNENO', 'LBDEONO', 'LBDBANO', 'LBXRBCSI', 'LBXHGB', 'LBXHCT',
    'LBXMCVSI', 'LBXMCHSI', 'LBXMC', 'LBXRDW', 'LBXPLTSI', 'LBXMPSI',
    'PHQ020', 'PHQ030', 'PHQ040', 'PHQ050', 'PHQ060', 'PHAFSTHR',
    'PHAFSTMN', 'PHDSESN', 'ACD010A', 'ALQ120Q', 'ALQ120U', 'ALQ130',
    'ALQ150', 'BPQ020', 'BPQ060', 'BPQ070', 'BPQ080', 'CDQ010',
    'DIQ010', 'DIQ050', 'HIQ210', 'HOD050', 'HOD060', 'HOQ065',
    'HOQ070', 'HOQ080', 'HSQ500', 'HSQ510', 'HSQ520', 'HSQ590',
    'HSAQUEX', 'HUQ010', 'HUQ020', 'HUQ030', 'HUQ040', 'HUQ050',
    'HUQ090', 'IMQ020', 'MCQ010', 'MCQ053', 'MCQ080', 'MCQ092',
    'MCQ140', 'MCQ160A', 'MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E',
    'MCQ160F', 'MCQ160G', 'MCQ160K', 'MCQ160L', 'MCQ220', 'OCQ260',
    'OCD270', 'OCD395', 'OSQ010A', 'OSQ010B', 'OSQ010C', 'OSQ060',
    'PAAQUEX', 'PFQ059', 'PFQ090', 'RDQ050', 'RDQ070', 'RDQ140',
    'SMQ020', 'SMD410', 'SMAQUEX', 'WHD010', 'WHD020', 'WHQ030',
    'WHD050', 'WHQ070', 'WHQ090', 'WHD110', 'WHD120', 'WHD140',
    'CDQ001', 'HSD010', 'HSQ470', 'HSQ480', 'HSQ490', 'KIQ022',
    'KIQ042', 'KIQ044', 'KIQ046', 'OCD150', 'OCD390G', 'SMQ680',
    'WHQ040', 'WHQ150', 'SIALANG', 'SIAPROXY', 'SIAINTRP', 'FIALANG',
    'FIAPROXY', 'FIAINTRP', 'MIALANG', 'MIAPROXY', 'MIAINTRP',
    'BMDSTATS', 'OHAPOS', 'LBDHDDSI', 'ALQ101', 'DBQ229', 'DBQ235A',
    'DBQ235B', 'DBQ235C', 'FSD032A', 'FSD032B', 'FSD032C', 'FSDHH',
    'FSD151', 'FSQ162', 'HSQ571', 'HUQ071', 'MCQ160M', 'PFQ049',
    'PFQ051', 'PFQ054', 'PFQ057'
]

# Map renamed variables back to raw NHANES names where needed
RENAMED_TO_RAW = {
    "age": "RIDAGEYR",
    "sex": "RIAGENDR",
    "race_ethnicity": "RIDRETH1",
    "education": "DMDEDUC2",
    "marital_status": "DMDMARTL",
    "poverty_income_ratio": "INDFMPIR",
    "household_income_cat": "INDHHIN2",
    "weight_interview_2yr": "WTINT2YR",
    "weight_exam_2yr": "WTMEC2YR",
}

LOOKUP_VARS = [RENAMED_TO_RAW.get(v, v) for v in TARGET_VARS if not v.startswith("death_")]


def parse_doc_file(doc_path: Path):
    with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    records = []
    value_records = []

    for div in tqdm(soup.find_all("div"), desc=f"Parsing {doc_path.name}"):
        title = div.find("h3", {"class": "vartitle"})
        info = div.find("dl")
        if title is None or info is None:
            continue

        var_name = title.get("id")
        if not var_name:
            continue

        info_dict = {}
        dts = info.find_all("dt")
        dds = info.find_all("dd")
        for dt, dd in zip(dts, dds):
            key = dt.get_text(" ", strip=True).strip(": ")
            val = dd.get_text(" ", strip=True)
            info_dict[key] = val

        record = {
            "variable_name": var_name.upper(),
            "sas_label": info_dict.get("SAS Label"),
            "english_text": info_dict.get("English Text"),
            "target": info_dict.get("Target"),
            "file_variable_name": info_dict.get("Variable Name"),
            "doc_path": str(doc_path),
        }
        records.append(record)

        table = div.find("table")
        if table is not None:
            try:
                code_df = pd.read_html(StringIO(str(table)))[0]
                code_df["variable_name"] = var_name.upper()
                code_df["doc_path"] = str(doc_path)
                value_records.append(code_df)
            except Exception:
                pass

    vars_df = pd.DataFrame(records)
    values_df = pd.concat(value_records, ignore_index=True) if value_records else pd.DataFrame()
    return vars_df, values_df


def collect_all_doc_lookups(raw_dir: Path):
    doc_files = sorted(raw_dir.glob("*/*/*.htm"))
    all_vars = []
    all_values = []

    for doc_file in tqdm(doc_files, desc="Collecting variable lookups"):
        try:
            vars_df, values_df = parse_doc_file(doc_file)
            parts = doc_file.parts

            if not vars_df.empty:
                cycle = parts[-3]
                component = parts[-2]
                vars_df["cycle"] = cycle
                vars_df["component"] = component
                all_vars.append(vars_df)

            if not values_df.empty:
                cycle = parts[-3]
                component = parts[-2]
                values_df["cycle"] = cycle
                values_df["component"] = component
                all_values.append(values_df)

        except Exception as e:
            print(f"[warn] Could not parse {doc_file}: {e}")

    if all_vars:
        variables_df = pd.concat(all_vars, ignore_index=True)
    else:
        variables_df = pd.DataFrame(columns=[
            "variable_name",
            "sas_label",
            "english_text",
            "target",
            "file_variable_name",
            "doc_path",
            "cycle",
            "component",
        ])

    if all_values:
        values_df = pd.concat(all_values, ignore_index=True)
    else:
        values_df = pd.DataFrame(columns=[
            "variable_name",
            "doc_path",
            "cycle",
            "component",
        ])

    return variables_df, values_df


def main():
    variables_df, values_df = collect_all_doc_lookups(RAW_DIR)

    variables_df.to_csv(OUT_DIR / "all_variable_descriptions.csv", index=False)
    values_df.to_csv(OUT_DIR / "all_variable_value_tables.csv", index=False)

    if "variable_name" not in variables_df.columns:
        raise RuntimeError(
            "No variable descriptions were parsed from the NHANES .htm files. "
            "Check whether the documentation files were downloaded correctly "
            "and whether parse_doc_file() matches their HTML structure."
        )

    selected = variables_df[variables_df["variable_name"].isin(LOOKUP_VARS)].copy()
    selected = selected.sort_values(["variable_name", "cycle", "component"])
    selected.to_csv(OUT_DIR / "selected_variable_descriptions.csv", index=False)

    if "variable_name" in values_df.columns:
        selected_values = values_df[values_df["variable_name"].isin(LOOKUP_VARS)].copy()
        selected_values = selected_values.sort_values(["variable_name", "cycle", "component"])
    else:
        selected_values = pd.DataFrame(columns=values_df.columns)

    selected_values.to_csv(OUT_DIR / "selected_variable_value_tables.csv", index=False)

    latest = (
        selected.sort_values(["variable_name", "cycle"])
        .groupby("variable_name", as_index=False)
        .tail(1)
        .sort_values("variable_name")
    )
    latest.to_csv(OUT_DIR / "selected_variable_descriptions_latest.csv", index=False)

    print("Saved:")
    print(OUT_DIR / "all_variable_descriptions.csv")
    print(OUT_DIR / "all_variable_value_tables.csv")
    print(OUT_DIR / "selected_variable_descriptions.csv")
    print(OUT_DIR / "selected_variable_value_tables.csv")
    print(OUT_DIR / "selected_variable_descriptions_latest.csv")


if __name__ == "__main__":
    main()