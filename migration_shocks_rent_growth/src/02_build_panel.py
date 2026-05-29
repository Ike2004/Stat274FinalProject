from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
INTERMEDIATE = ROOT / "data" / "intermediate"


def county_fips(state, county) -> str:
    if pd.isna(state) or pd.isna(county):
        return np.nan
    return f"{int(state):02d}{int(county):03d}"


def build_zori() -> pd.DataFrame:
    files = list((RAW / "zillow").glob("*zori*.csv")) + list((RAW / "zillow").glob("*ZORI*.csv"))
    if not files:
        raise FileNotFoundError("No ZORI CSV found in data/raw/zillow/. Run 01_download_data.py or download county ZORI manually.")

    df = pd.read_csv(files[0])
    date_cols = [c for c in df.columns if re.match(r"^\d{4}-\d{2}-\d{2}$", str(c))]
    if not date_cols:
        raise ValueError("Could not identify monthly date columns in ZORI file.")

    if {"StateCodeFIPS", "MunicipalCodeFIPS"}.issubset(df.columns):
        df["fips"] = [county_fips(s, c) for s, c in zip(df["StateCodeFIPS"], df["MunicipalCodeFIPS"])]
    elif "RegionName" in df.columns and df["RegionName"].astype(str).str.fullmatch(r"\d{5}").any():
        df["fips"] = df["RegionName"].astype(str).str.zfill(5)
    else:
        raise ValueError("Could not construct county FIPS from ZORI file.")

    long = df[["fips"] + date_cols].melt("fips", var_name="month", value_name="rent")
    long["month"] = pd.to_datetime(long["month"])
    long["year"] = long["month"].dt.year
    annual = (
        long.dropna(subset=["rent"])
        .groupby(["fips", "year"], as_index=False)
        .agg(rent=("rent", "mean"), rent_months=("rent", "size"))
    )
    annual = annual.sort_values(["fips", "year"])
    annual["log_rent"] = np.log(annual["rent"])
    annual["d_log_rent"] = annual.groupby("fips")["log_rent"].diff()
    return annual


def read_any_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python")


def standardize_migration_file(path: Path, year: int | None) -> pd.DataFrame | None:
    df = read_any_table(path)
    cols = {c.lower().strip(): c for c in df.columns}

    if {"fips", "net_migration"}.issubset(cols):
        out = df.rename(columns={cols["fips"]: "fips", cols["net_migration"]: "net_migration"}).copy()
        out["year"] = out[cols.get("year", "year")] if "year" in cols else year
        return out[["fips", "year", "net_migration"]]

    # IRS public files commonly distinguish origin-year and destination-year FIPS fields.
    dest_state = next((cols[k] for k in cols if k in {"y2_statefips", "dest_statefips", "statefips"}), None)
    dest_county = next((cols[k] for k in cols if k in {"y2_countyfips", "dest_countyfips", "countyfips"}), None)
    people = next((cols[k] for k in cols if k in {"n2", "exemptions", "individuals"}), None)
    if not (dest_state and dest_county and people):
        return None

    tmp = df[[dest_state, dest_county, people]].copy()
    tmp["fips"] = [county_fips(s, c) for s, c in zip(tmp[dest_state], tmp[dest_county])]
    tmp["year"] = year
    tmp["gross_inflow"] = pd.to_numeric(tmp[people], errors="coerce")
    return tmp.groupby(["fips", "year"], as_index=False).agg(gross_inflow=("gross_inflow", "sum"))


def infer_year(path: Path) -> int | None:
    compact = re.search(r"(18|19|20|21|22)(19|20|21|22|23)", path.name)
    if compact:
        return 2000 + int(compact.group(2))
    m = re.search(r"(20\d{2})[-_ ]?(?:to|-)?[-_ ]?(20\d{2})", path.name)
    if m:
        return int(m.group(2))
    m = re.search(r"(20\d{2})", path.name)
    return int(m.group(1)) if m else None


def build_migration() -> pd.DataFrame:
    frames = []
    for path in (RAW / "irs").glob("countyinflow*.csv"):
        year = infer_year(path)
        df = pd.read_csv(path, encoding="latin1")
        if year is None:
            continue
        # Total Migration-US is y1_statefips=97, y1_countyfips=0 in IRS inflow files.
        total = df[(df["y1_statefips"] == 97) & (df["y1_countyfips"] == 0)].copy()
        total["fips"] = [county_fips(s, c) for s, c in zip(total["y2_statefips"], total["y2_countyfips"])]
        total["year"] = year
        total["gross_inflow"] = pd.to_numeric(total["n2"], errors="coerce")
        frames.append(total[["fips", "year", "gross_inflow"]])

    for path in (RAW / "irs").glob("countyoutflow*.csv"):
        year = infer_year(path)
        df = pd.read_csv(path, encoding="latin1")
        if year is None:
            continue
        # Total Migration-US is y2_statefips=97, y2_countyfips=0 in IRS outflow files.
        total = df[(df["y2_statefips"] == 97) & (df["y2_countyfips"] == 0)].copy()
        total["fips"] = [county_fips(s, c) for s, c in zip(total["y1_statefips"], total["y1_countyfips"])]
        total["year"] = year
        total["gross_outflow"] = pd.to_numeric(total["n2"], errors="coerce")
        frames.append(total[["fips", "year", "gross_outflow"]])

        nonmigrants = df[
            (df["y2_statefips"] == df["y1_statefips"])
            & (df["y2_countyfips"] == df["y1_countyfips"])
        ].copy()
        nonmigrants["fips"] = [
            county_fips(s, c) for s, c in zip(nonmigrants["y1_statefips"], nonmigrants["y1_countyfips"])
        ]
        nonmigrants["year"] = year
        nonmigrants["nonmigrant_exemptions"] = pd.to_numeric(nonmigrants["n2"], errors="coerce")
        frames.append(nonmigrants[["fips", "year", "nonmigrant_exemptions"]])

    if not frames:
        print("No parseable IRS migration files found. Continuing without migration variables.")
        return pd.DataFrame(columns=["fips", "year", "gross_inflow", "gross_outflow", "net_migration"])

    mig = pd.concat(frames, ignore_index=True)
    mig = mig.groupby(["fips", "year"], as_index=False).sum(numeric_only=True)
    mig["net_migration"] = mig.get("gross_inflow", 0) - mig.get("gross_outflow", 0)
    mig["irs_population_base"] = mig.get("nonmigrant_exemptions", 0) + mig.get("gross_outflow", 0)
    return mig


def add_optional_csv(panel: pd.DataFrame, folder: str, filename: str) -> pd.DataFrame:
    path = RAW / folder / filename
    if not path.exists():
        return panel
    add = pd.read_csv(path)
    add["fips"] = add["fips"].astype(str).str.zfill(5)
    return panel.merge(add, on=[c for c in ["fips", "year"] if c in add.columns], how="left")


def build_bps_2020() -> pd.DataFrame:
    path = RAW / "bps" / "bps_2020_county.txt"
    if not path.exists():
        print("No 2020 BPS county file found; continuing without permits proxy.")
        return pd.DataFrame(columns=["fips", "permits_2020_units"])

    df = pd.read_csv(path, skiprows=3, header=None, encoding="latin1")
    df = df[df[0].astype(str).str.fullmatch(r"\d{4}", na=False)].copy()
    df["fips"] = df[1].astype(str).str.zfill(2) + df[2].astype(str).str.zfill(3)
    unit_cols = [7, 10, 13, 16]
    for col in unit_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["permits_2020_units"] = df[unit_cols].sum(axis=1)
    return df[["fips", "permits_2020_units"]]


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)

    zori = build_zori()
    zori.to_csv(INTERMEDIATE / "zori_county_year.csv", index=False)

    mig = build_migration()
    panel = zori.merge(mig, on=["fips", "year"], how="left")
    panel = add_optional_csv(panel, "acs", "acs_county_year.csv")
    panel = panel.merge(build_bps_2020(), on="fips", how="left")
    panel = add_optional_csv(panel, "bps", "bps_county_year.csv")

    if "population" in panel.columns:
        panel = panel.sort_values(["fips", "year"])
        panel["population_lag"] = panel.groupby("fips")["population"].shift(1)
        if "net_migration" in panel.columns:
            panel["net_mig_share"] = panel["net_migration"] / panel["population_lag"]
        if "gross_inflow" in panel.columns:
            panel["gross_inflow_share"] = panel["gross_inflow"] / panel["population_lag"]
    elif "irs_population_base" in panel.columns:
        panel["population_lag"] = panel["irs_population_base"]
        panel["net_mig_share"] = panel["net_migration"] / panel["population_lag"]
        panel["gross_inflow_share"] = panel["gross_inflow"] / panel["population_lag"]

    base_2020 = (
        panel.loc[panel["year"].eq(2020), ["fips", "irs_population_base"]]
        .dropna()
        .rename(columns={"irs_population_base": "irs_population_base_2020"})
    )
    panel = panel.merge(base_2020, on="fips", how="left")
    panel["permits_2020_per_capita"] = panel["permits_2020_units"] / panel["irs_population_base_2020"]

    panel.to_csv(PROCESSED / "county_year_panel.csv", index=False)
    print(f"wrote {PROCESSED / 'county_year_panel.csv'} with {len(panel):,} rows")


if __name__ == "__main__":
    main()
