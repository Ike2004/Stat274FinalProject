from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

ZILLOW_DATA_PAGE = "https://www.zillow.com/research/data/"
IRS_MIGRATION_PAGE = "https://www.irs.gov/statistics/soi-tax-stats-migration-data"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists: {dest}")
        return
    print(f"downloading: {url}")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"wrote: {dest}")


def scrape_links(url: str) -> list[str]:
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    return [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]


def download_zori() -> None:
    links = scrape_links(ZILLOW_DATA_PAGE)
    zori_links = [
        link for link in links
        if "files.zillowstatic.com" in link
        and "zori" in link.lower()
        and "county" in link.lower()
        and link.lower().endswith(".csv")
    ]
    if not zori_links:
        print("No county ZORI CSV discovered automatically. Download it from Zillow Research and place it in data/raw/zillow/.")
        return
    # Prefer seasonally adjusted all-homes-plus-multifamily if present; otherwise take first county ZORI.
    zori_links = sorted(zori_links, key=lambda x: ("sfr" in x.lower(), "mf" in x.lower(), x))
    url = zori_links[0]
    dest = RAW / "zillow" / Path(url.split("?")[0]).name
    download(url, dest)


def download_irs_pages() -> None:
    links = scrape_links(IRS_MIGRATION_PAGE)
    year_links = [
        link for link in links
        if re.search(r"/soi-tax-stats-migration-data-\d{4}-\d{4}", link)
        or re.search(r"\d{4}\s+to\s+\d{4}", link, re.I)
    ]
    (RAW / "irs").mkdir(parents=True, exist_ok=True)
    with (RAW / "irs" / "source_links.txt").open("w") as f:
        for link in sorted(set(year_links)):
            f.write(link + "\n")
    print(f"wrote IRS source link inventory: {RAW / 'irs' / 'source_links.txt'}")
    print("IRS file layouts vary by year; download county inflow/outflow CSV or ZIP files listed there into data/raw/irs/.")


def unzip_archives(folder: Path) -> None:
    for zip_path in folder.glob("*.zip"):
        out_dir = zip_path.with_suffix("")
        if out_dir.exists():
            continue
        print(f"extracting: {zip_path}")
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    download_zori()
    download_irs_pages()
    unzip_archives(RAW / "irs")
    print("Done. Add ACS, BPS, and zoning files as described in README if they are not downloaded separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

