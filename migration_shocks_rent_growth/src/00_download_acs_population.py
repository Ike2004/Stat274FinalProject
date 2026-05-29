from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "acs" / "acs_county_year.csv"


def fetch_year(year: int) -> pd.DataFrame:
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME,B01003_001E",
        "for": "county:*",
        "in": "state:*",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["year"] = year
    df["fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    df["population"] = pd.to_numeric(df["B01003_001E"], errors="coerce")
    return df[["fips", "year", "population"]]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel = pd.concat([fetch_year(year) for year in range(2018, 2024)], ignore_index=True)
    panel.to_csv(OUT, index=False)
    print(f"wrote {OUT} with {len(panel):,} rows")


if __name__ == "__main__":
    main()
