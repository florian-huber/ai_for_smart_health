from pathlib import Path
import re
import io
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path("nhanes_components")
RAW_DIR = BASE_DIR / "raw"
META_DIR = BASE_DIR / "metadata"
MERGED_DIR = BASE_DIR / "merged"

for d in [RAW_DIR, META_DIR, MERGED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CYCLES = [
    "1999-2000",
    "2001-2002",
    "2003-2004",
    "2005-2006",
    "2007-2008",
    "2009-2010",
    "2011-2012",
    "2013-2014",
]

COMPONENTS = [
    "Demographics",
    "Examination",
    "Laboratory",
    "Questionnaire",
]

EXCLUDE_IF_DESCRIPTION_CONTAINS = [
    "Raw Data",
    "Minute",
]

EXCLUDE_IF_LABEL_CONTAINS = [
    "FTP",
]

SKIP_FILE_PREFIXES = {
    "PAXMIN",
    "PAX80",
    "PAXLUX",
    "PAXHR",
    "SPXRAW",
}

BASE_SEARCH_URL = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx"
BASE_SITE = "https://wwwn.cdc.gov"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; NHANES-component-collector/1.0)"
})

# True --> build merged per-cycle tables
BUILD_MERGED_TABLES = True

# True --> save doc pages
DOWNLOAD_DOC_PAGES = True

# A short pause to be polite to the CDC servers
REQUEST_SLEEP_SECONDS = 0.25


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def safe_get(url: str, timeout: int = 120) -> requests.Response:
    time.sleep(REQUEST_SLEEP_SECONDS)
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def ensure_binary_not_html(content: bytes, content_type: str, url: str) -> None:
    start = content[:200].lower()
    if b"<html" in start or "text/html" in content_type.lower():
        raise RuntimeError(
            f"Expected binary file but got HTML.\n"
            f"URL: {url}\n"
            f"Content-Type: {content_type}\n"
            f"First bytes: {content[:120]!r}"
        )


def download_file(url: str, dest: Path, expect_binary: bool = True) -> Path:
    if dest.exists():
        print(f"[skip] {dest}")
        return dest

    print(f"[download] {url}")
    r = safe_get(url)
    if expect_binary:
        ensure_binary_not_html(r.content, r.headers.get("Content-Type", ""), url)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest


def read_xpt(path: Path) -> pd.DataFrame:
    print(f"[read xpt] {path}")
    return pd.read_sas(path, format="xport")


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def make_absolute_url(href: str, page_url: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return requests.compat.urljoin(page_url, href)


# -----------------------------------------------------------------------------
# Scraping NHANES component pages
# -----------------------------------------------------------------------------

def component_page_url(component: str) -> str:
    return f"{BASE_SEARCH_URL}?Component={component}"


def scrape_component_page(component: str) -> pd.DataFrame:
    """
    Scrape the NHANES component page and return a manifest of file entries.

    Expected output columns:
      cycle
      component
      data_file_code
      data_file_description
      doc_url
      data_url
      doc_label
      data_label
      date_published
    """
    page_url = component_page_url(component)
    print(f"[open page] {page_url}")

    r = safe_get(page_url)
    soup = BeautifulSoup(r.text, "html.parser")

    text = soup.get_text("\n")
    if "Years Data File Name Doc File Data File Date Published" not in text:
        print(f"[warning] Could not find expected table header text on {page_url}")

    records = []

    # The table rows contain cycle text and two relevant anchors: Doc + Data [XPT]
    # We scan all anchors and group doc/data links by nearby cycle text.
    rows = soup.find_all("tr")

    for row in rows:
        row_text = " ".join(row.stripped_strings)

        cycle_match = re.search(r"\b(1999-2000|2001-2002|2003-2004|2005-2006|2007-2008|2009-2010|2011-2012|2013-2014)\b", row_text)
        if not cycle_match:
            continue

        cycle = cycle_match.group(1)
        anchors = row.find_all("a", href=True)

        doc_url = None
        data_url = None
        doc_label = None
        data_label = None
        data_file_code = None

        for a in anchors:
            href = a["href"]
            label = " ".join(a.stripped_strings)

            if ".htm" in href.lower():
                doc_url = make_absolute_url(href, page_url)
                doc_label = label

            if ".xpt" in href.lower():
                data_url = make_absolute_url(href, page_url)
                data_label = label

                # extract file code from the URL, e.g. DEMO_H, BPX_D, ALB_CR_H
                m = re.search(r"/([A-Za-z0-9_]+)\.xpt$", href, flags=re.IGNORECASE)
                if m:
                    data_file_code = m.group(1).upper()

        if data_url is None:
            continue

        # Try to recover a readable description from row text
        description = row_text
        if doc_label:
            description = description.replace(doc_label, "").strip()
        if data_label:
            description = description.replace(data_label, "").strip()

        records.append({
            "cycle": cycle,
            "component": component,
            "data_file_code": data_file_code,
            "data_file_description": description,
            "doc_url": doc_url,
            "data_url": data_url,
            "doc_label": doc_label,
            "data_label": data_label,
            "page_url": page_url,
        })

    if not records:
        raise RuntimeError(f"No file records found for component={component}")

    df = pd.DataFrame(records).drop_duplicates()
    df = df.sort_values(["cycle", "data_file_code"]).reset_index(drop=True)
    return df


def build_component_manifest(components=None) -> pd.DataFrame:
    if components is None:
        components = COMPONENTS

    frames = []
    for component in components:
        df = scrape_component_page(component)
        frames.append(df)

    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest[manifest["cycle"].isin(CYCLES)].copy()
    manifest = manifest.sort_values(["cycle", "component", "data_file_code"]).reset_index(drop=True)
    return manifest


# -----------------------------------------------------------------------------
# Downloading files
# -----------------------------------------------------------------------------



def should_skip_file(file_code: str) -> bool:
    if not isinstance(file_code, str):
        return False
    file_code = file_code.upper()
    return any(file_code.startswith(prefix) for prefix in SKIP_FILE_PREFIXES)


def download_component_files(manifest: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()

    local_xpt_paths = []
    local_doc_paths = []

    for _, row in manifest.iterrows():
        cycle = row["cycle"]
        component = row["component"]
        file_code = row["data_file_code"]
        data_url = row["data_url"]
        doc_url = row["doc_url"]

        if should_skip_file(file_code):
            print(f"[skip large/raw] {file_code}")
            local_xpt_paths.append(None)
            local_doc_paths.append(None)
            continue

        cycle_dir = RAW_DIR / cycle / component
        xpt_name = sanitize_filename(f"{file_code}.XPT")
        xpt_path = cycle_dir / xpt_name
        download_file(data_url, xpt_path, expect_binary=True)
        local_xpt_paths.append(str(xpt_path))

        if DOWNLOAD_DOC_PAGES and isinstance(doc_url, str) and doc_url:
            doc_name = sanitize_filename(f"{file_code}.htm")
            doc_path = cycle_dir / doc_name
            download_file(doc_url, doc_path, expect_binary=False)
            local_doc_paths.append(str(doc_path))
        else:
            local_doc_paths.append(None)

    manifest["local_xpt_path"] = local_xpt_paths
    manifest["local_doc_path"] = local_doc_paths
    return manifest


# -----------------------------------------------------------------------------
# Catalog variables in each XPT file
# -----------------------------------------------------------------------------
def catalog_xpt_variables(manifest: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in manifest.iterrows():
        xpt_path_str = row["local_xpt_path"]
        if not xpt_path_str:
            continue

        xpt_path = Path(xpt_path_str)
        cycle = row["cycle"]
        component = row["component"]
        file_code = row["data_file_code"]

        try:
            df = read_xpt(xpt_path)
        except Exception as e:
            print(f"[error] Could not read {xpt_path}: {e}")
            continue

        for col in df.columns:
            records.append({
                "cycle": cycle,
                "component": component,
                "data_file_code": file_code,
                "variable_name": str(col),
                "dtype": str(df[col].dtype),
                "n_rows": len(df),
            })

    catalog = pd.DataFrame(records)
    return catalog.sort_values(
        ["cycle", "component", "data_file_code", "variable_name"]
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Optional: merge all files within a cycle
# -----------------------------------------------------------------------------

def rename_non_seqn_columns(df: pd.DataFrame, file_code: str) -> pd.DataFrame:
    """
    Rename columns to keep provenance and avoid collisions:
      RIDAGEYR -> RIDAGEYR__DEMO_H
    while keeping SEQN unchanged.
    """
    rename_map = {}
    for col in df.columns:
        if col != "SEQN":
            rename_map[col] = f"{col}__{file_code}"
    return df.rename(columns=rename_map)


def merge_cycle_files(manifest_cycle: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all files for one cycle on SEQN using outer joins.
    Keeps provenance by suffixing each variable with __FILECODE.
    """
    merged = None

    for _, row in manifest_cycle.iterrows():
        xpt_path_str = row["local_xpt_path"]
        file_code = row["data_file_code"]

        # Skip files that were intentionally excluded
        if not xpt_path_str:
            print(f"[skip missing path] {file_code}")
            continue

        xpt_path = Path(xpt_path_str)

        try:
            df = read_xpt(xpt_path)
        except Exception as e:
            print(f"[error] Skipping unreadable file {xpt_path}: {e}")
            continue

        if "SEQN" not in df.columns:
            print(f"[warning] Skipping file without SEQN: {xpt_path}")
            continue

        df = rename_non_seqn_columns(df, file_code)

        if merged is None:
            merged = df.copy()
        else:
            merged = merged.merge(df, on="SEQN", how="outer")

    if merged is None:
        raise RuntimeError("No mergeable files found for cycle.")

    return merged


def build_merged_cycle_tables(manifest: pd.DataFrame) -> None:
    for cycle in sorted(manifest["cycle"].unique()):
        print(f"\n[merge cycle] {cycle}")
        manifest_cycle = manifest[
            (manifest["cycle"] == cycle) &
            (manifest["local_xpt_path"].notna())
        ].copy()

        merged = merge_cycle_files(manifest_cycle)

        try:
            out_path = MERGED_DIR / f"nhanes_{cycle}_all_components.parquet"
            merged.to_parquet(out_path, index=False)
        except ImportError:
            out_path = MERGED_DIR / f"nhanes_{cycle}_all_components.csv"
            merged.to_csv(out_path, index=False)

        print(f"[saved] {out_path} shape={merged.shape}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("\n=== Step 1: Build component manifest ===")
    manifest = build_component_manifest(COMPONENTS)
    manifest_path = META_DIR / "component_file_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[saved] {manifest_path}")
    print(manifest.groupby(["cycle", "component"]).size())

    print("\n=== Step 2: Download files ===")
    manifest = download_component_files(manifest)
    manifest_downloaded_path = META_DIR / "component_file_manifest_downloaded.csv"
    manifest.to_csv(manifest_downloaded_path, index=False)
    print(f"[saved] {manifest_downloaded_path}")

    print("\n=== Step 3: Build variable catalog ===")
    variable_catalog = catalog_xpt_variables(manifest)
    variable_catalog_path = META_DIR / "variable_catalog.csv"
    variable_catalog.to_csv(variable_catalog_path, index=False)
    print(f"[saved] {variable_catalog_path}")
    print(f"[info] Variable catalog rows: {len(variable_catalog):,}")

    # Helpful summary: in how many cycles does each variable appear?
    variable_presence = (
        variable_catalog.groupby(["component", "variable_name"])["cycle"]
        .nunique()
        .reset_index(name="n_cycles_present")
        .sort_values(["n_cycles_present", "component", "variable_name"], ascending=[False, True, True])
    )
    variable_presence_path = META_DIR / "variable_presence_summary.csv"
    variable_presence.to_csv(variable_presence_path, index=False)
    print(f"[saved] {variable_presence_path}")

    if BUILD_MERGED_TABLES:
        print("\n=== Step 4: Build merged per-cycle tables ===")
        build_merged_cycle_tables(manifest)

    print("\nDone.")
    print("\nKey outputs:")
    print(f" - {manifest_downloaded_path}")
    print(f" - {variable_catalog_path}")
    print(f" - {variable_presence_path}")
    if BUILD_MERGED_TABLES:
        print(f" - {MERGED_DIR}")


if __name__ == "__main__":
    main()