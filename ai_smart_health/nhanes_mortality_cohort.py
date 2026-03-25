from pathlib import Path
import io
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path("nhanes_mortality_data")
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NHANES cycles used in the paper
CYCLES = [
    ("1999-2000", ""),
    ("2001-2002", "_B"),
    ("2003-2004", "_C"),
    ("2005-2006", "_D"),
    ("2007-2008", "_E"),
    ("2009-2010", "_F"),
    ("2011-2012", "_G"),
    ("2013-2014", "_H"),
]

# Official CDC endpoints
NHANES_XPT_BASE = "https://wwwn.cdc.gov/Nchs/Nhanes"
MORTALITY_2019_BASE = "https://ftp.cdc.gov/pub/health_statistics/NCHS/datalinkage/linked_mortality"

# Theoreticall possible to use local data (False means this will be skipped)
USE_LOCAL_2015_MORTALITY = False
LOCAL_2015_MORTALITY_DIR = RAW_DIR / "mortality_2015"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def download_file(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return dest

    print(f"[download] {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; NHANES-cohort-builder/1.0)"
})


def get_nhanes_datapage_url(cycle: str, component: str = "Demographics") -> str:
    """
    Official NHANES dataset listing page for one cycle/component.
    Example:
    https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&Cycle=1999-2000
    """
    return (
        "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx"
        f"?Component={component}&Cycle={cycle}"
    )


def get_xpt_link_from_datapage(cycle: str, file_stem: str, component: str = "Demographics") -> str:
    """
    Scrape the official NHANES data page and extract the real XPT download link.

    file_stem examples:
      DEMO
      BMX
      BPX
      ALQ
    """
    page_url = get_nhanes_datapage_url(cycle, component)
    print(f"[open page] {page_url}")

    r = SESSION.get(page_url, timeout=120)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Look for any anchor whose href ends with the desired file, e.g. DEMO.xpt or DEMO_H.xpt
    anchors = soup.find_all("a", href=True)

    pattern = re.compile(rf"/{re.escape(file_stem)}(?:_[A-Z])?\.xpt$", re.IGNORECASE)

    matches = []
    for a in anchors:
        href = a["href"]
        if pattern.search(href):
            if href.startswith("http"):
                matches.append(href)
            else:
                matches.append(requests.compat.urljoin(page_url, href))

    if not matches:
        # Helpful debugging output
        candidate_links = [a["href"] for a in anchors if ".xpt" in a["href"].lower()]
        raise RuntimeError(
            f"Could not find XPT link for {file_stem} in cycle {cycle}.\n"
            f"Page: {page_url}\n"
            f"Found XPT-like links: {candidate_links[:10]}"
        )

    # Usually exactly one match
    xpt_url = matches[0]
    print(f"[xpt link] {xpt_url}")
    return xpt_url


def download_binary(url: str, dest: Path) -> Path:
    """
    Download a file and make sure we did not accidentally receive HTML.
    """
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return dest

    print(f"[download] {url}")
    r = SESSION.get(url, timeout=120)
    r.raise_for_status()

    # Guard against HTML error pages masquerading as downloads
    content_start = r.content[:200].lower()
    content_type = r.headers.get("Content-Type", "").lower()

    if b"<html" in content_start or "text/html" in content_type:
        raise RuntimeError(
            f"Expected binary XPT file but got HTML.\n"
            f"URL: {url}\n"
            f"Content-Type: {content_type}\n"
            f"First bytes: {r.content[:120]!r}"
        )

    dest.write_bytes(r.content)
    return dest


def read_xpt_file(path: Path) -> pd.DataFrame:
    """
    Read a local SAS XPORT file.
    """
    print(f"[read xpt] {path}")
    return pd.read_sas(path, format="xport")


def load_nhanes_file(cycle: str, file_stem: str, component: str, raw_dir: Path) -> pd.DataFrame:
    """
    Generic loader for one NHANES file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    xpt_url = get_xpt_link_from_datapage(cycle=cycle, file_stem=file_stem, component=component)

    # preserve the true filename from the URL
    filename = xpt_url.rstrip("/").split("/")[-1]
    local_path = raw_dir / filename

    download_binary(xpt_url, local_path)
    df = read_xpt_file(local_path)
    return df

def read_xpt_from_file(path: Path) -> pd.DataFrame:
    print(f"[read xpt] {path}")
    return pd.read_sas(path, format="xport")


def read_mortality_fixed_width(path: Path) -> pd.DataFrame:
    """
    NHANES public-use linked mortality fixed-width layout.
    Based on the NCHS sample read-in program for NHANES public-use mortality files.

    Columns:
      publicid   1-14
      eligstat  15
      mortstat  16
      ucod_leading 17-19
      diabetes  20
      hyperten  21
      dodqtr    22
      dodyear   23-26
      wgt_new   27-34   (NHIS-oriented, not used for NHANES)
      sa_wgt_new 35-42  (NHIS-oriented, not used for NHANES)
      permth_int 43-45
      permth_exm 46-48
    """
    colspecs = [
        (0, 14),   # publicid
        (14, 15),  # eligstat
        (15, 16),  # mortstat
        (16, 19),  # ucod_leading
        (19, 20),  # diabetes
        (20, 21),  # hyperten
        (21, 22),  # dodqtr
        (22, 26),  # dodyear
        (26, 34),  # wgt_new
        (34, 42),  # sa_wgt_new
        (42, 45),  # permth_int
        (45, 48),  # permth_exm
    ]
    names = [
        "publicid",
        "eligstat",
        "mortstat",
        "ucod_leading",
        "diabetes_flag",
        "hypertension_flag",
        "dodqtr",
        "dodyear",
        "wgt_new",
        "sa_wgt_new",
        "permth_int",
        "permth_exm",
    ]

    df = pd.read_fwf(path, colspecs=colspecs, names=names, na_values=".")
    # For NHANES, SEQN is the first 5 characters of PUBLICID in the sample program
    df["SEQN"] = pd.to_numeric(df["publicid"].astype(str).str[:5], errors="coerce")
    df = df.drop(columns=["publicid", "dodqtr", "dodyear", "wgt_new", "sa_wgt_new"])
    return df


def mortality_filename(cycle: str, release_year: int = 2019) -> str:
    cycle_us = cycle.replace("-", "_")
    return f"NHANES_{cycle_us}_MORT_{release_year}_PUBLIC.dat"


def demo_filename(suffix: str) -> str:
    # 1999-2000 uses DEMO.XPT, later cycles DEMO_B.XPT, ..., DEMO_H.XPT
    return f"DEMO{suffix}.XPT"


def demo_url(cycle: str, suffix: str) -> str:
    return f"{NHANES_XPT_BASE}/{cycle}/{demo_filename(suffix)}"


def mortality_url_2019(cycle: str) -> str:
    return f"{MORTALITY_2019_BASE}/{mortality_filename(cycle, 2019)}"


# -----------------------------------------------------------------------------
# Cycle loaders
# -----------------------------------------------------------------------------

def load_demo_cycle(cycle: str, raw_dir) -> pd.DataFrame:
    raw_dir = Path(raw_dir)

    df = load_nhanes_file(
        cycle=cycle,
        file_stem="DEMO",
        component="Demographics",
        raw_dir=raw_dir / cycle
    )

    cols = [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH1",
        "DMDEDUC2",
        "DMDMARTL",
        "INDHHIN2",
        "INDFMPIR",
        "WTINT2YR",
        "WTMEC2YR",
        "SDMVPSU",
        "SDMVSTRA",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols].copy()
    df["cycle"] = cycle
    return df


def load_mortality_cycle(cycle: str) -> pd.DataFrame:
    """
    Load linked mortality file for one cycle.

    By default this uses the current public 2019 release because the URLs are stable.
    To mirror the paper exactly, place the archived 2015 .dat files locally and set
    USE_LOCAL_2015_MORTALITY = True.
    """
    if USE_LOCAL_2015_MORTALITY:
        path = LOCAL_2015_MORTALITY_DIR / mortality_filename(cycle, 2015)
        if not path.exists():
            raise FileNotFoundError(
                f"Expected archived 2015 mortality file not found: {path}"
            )
        df = read_mortality_fixed_width(path)
    else:
        # Stable public file layout with same key variables used here
        path = RAW_DIR / mortality_filename(cycle, 2019)
        download_file(mortality_url_2019(cycle), path)
        df = read_mortality_fixed_width(path)

    df["cycle"] = cycle
    return df


def build_cycle_dataset(cycle: str, suffix: str) -> pd.DataFrame:
    demo = load_demo_cycle(cycle, suffix)
    mort = load_mortality_cycle(cycle)

    df = demo.merge(mort, on=["SEQN", "cycle"], how="left")

    # Adults only: public linked mortality uses ELIGSTAT to flag under-18 participants
    # ELIGSTAT = 1 eligible, 2 under age 18, 3 ineligible due to insufficient identifying data
    df = df[df["eligstat"] == 1].copy()

    # Keep consistency with the paper:
    # topcode all ages >= 80 to 80
    if "RIDAGEYR" in df.columns:
        df["RIDAGEYR"] = df["RIDAGEYR"].clip(upper=80)

    # More readable names
    rename_map = {
        "RIDAGEYR": "age",
        "RIAGENDR": "sex",
        "RIDRETH1": "race_ethnicity",
        "DMDEDUC2": "education",
        "DMDMARTL": "marital_status",
        "INDHHIN2": "household_income_cat",
        "INDFMPIR": "poverty_income_ratio",
        "WTINT2YR": "weight_interview_2yr",
        "WTMEC2YR": "weight_exam_2yr",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Mortality labels for common horizons
    # MORTSTAT: 0 assumed alive, 1 assumed deceased
    # PERMTH_INT: person-months from interview to death or censoring
    df["death_1yr"] = ((df["mortstat"] == 1) & (df["permth_int"] <= 12)).astype(int)
    df["death_3yr"] = ((df["mortstat"] == 1) & (df["permth_int"] <= 36)).astype(int)
    df["death_5yr"] = ((df["mortstat"] == 1) & (df["permth_int"] <= 60)).astype(int)
    df["death_10yr"] = ((df["mortstat"] == 1) & (df["permth_int"] <= 120)).astype(int)

    return df


def build_nhanes_mortality_cohort() -> pd.DataFrame:
    frames = []
    for cycle, suffix in CYCLES:
        print(f"\n=== Processing {cycle} ===")
        cycle_df = build_cycle_dataset(cycle, suffix)
        frames.append(cycle_df)

    df = pd.concat(frames, ignore_index=True)

    # Ensure numeric SEQN
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")

    return df


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cohort = build_nhanes_mortality_cohort()

    print("\nFinal shape:", cohort.shape)
    print("\nColumns:")
    print(cohort.columns.tolist())

    print("\nMortality label prevalence:")
    for col in ["death_1yr", "death_3yr", "death_5yr", "death_10yr"]:
        print(col, cohort[col].mean())

    print("\nEligibility / mortality status summary:")
    print(cohort[["cycle", "mortstat"]].value_counts(dropna=False).sort_index())

    cohort.to_csv(OUT_DIR / "nhanes_1999_2014_demo_mortality_cohort.csv", index=False)
    print(f"\nSaved to: {OUT_DIR / 'nhanes_1999_2014_demo_mortality_cohort.csv'}")