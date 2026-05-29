from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
TABLES = ROOT / "output" / "tables"


def add_treatment_bins(df: pd.DataFrame, shock_col: str = "net_mig_share") -> pd.DataFrame:
    df = df.copy()
    if shock_col not in df.columns:
        raise ValueError(f"Missing {shock_col}. Add population and migration data before estimating models.")

    shock = (
        df.loc[df["year"].between(2020, 2022)]
        .groupby("fips", as_index=False)
        .agg(pandemic_shock=(shock_col, "sum"))
    )
    q25, q75 = shock["pandemic_shock"].quantile([0.25, 0.75])
    shock["high_inflow"] = (shock["pandemic_shock"] >= q75).astype(int)
    shock["low_inflow"] = (shock["pandemic_shock"] <= q25).astype(int)
    shock["neutral_inflow"] = (shock["pandemic_shock"].abs() <= 0.005).astype(int)
    out = df.merge(shock, on="fips", how="left")
    out["post"] = (out["year"] >= 2020).astype(int)
    return out[out["high_inflow"].eq(1) | out["low_inflow"].eq(1) | out["neutral_inflow"].eq(1)].copy()


def add_zoning(df: pd.DataFrame) -> pd.DataFrame:
    path = RAW / "zoning" / "zoning_county.csv"
    if not path.exists():
        return df
    zoning = pd.read_csv(path)
    zoning["fips"] = zoning["fips"].astype(str).str.zfill(5)
    out = df.merge(zoning, on="fips", how="left")
    if "zoning_index" in out.columns:
        med = out[["fips", "zoning_index"]].drop_duplicates()["zoning_index"].median()
        out["high_zoning"] = (out["zoning_index"] >= med).astype(float)
    return out


def fit(formula: str, df: pd.DataFrame):
    terms = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula))
    ignore = {"C"}
    needed = ["d_log_rent"] + sorted(t for t in terms if t in df.columns and t not in ignore)
    model_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    res = smf.ols(formula, data=model_df).fit(cov_type="cluster", cov_kwds={"groups": model_df["fips"]})
    return res


def write_result(path: Path, title: str, res) -> None:
    with path.open("w") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        f.write(res.summary().as_text())


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PROCESSED / "county_year_panel.csv")
    panel["fips"] = panel["fips"].astype(str).str.zfill(5)
    df = add_zoning(add_treatment_bins(panel))

    controls = []
    for c in ["median_income", "unemployment_rate", "renter_share", "vacancy_rate", "permits_per_capita"]:
        if c in df.columns:
            controls.append(c)
    rhs_controls = " + " + " + ".join(controls) if controls else ""

    baseline = fit(f"d_log_rent ~ high_inflow:post + low_inflow:post{rhs_controls} + C(fips) + C(year)", df)
    write_result(TABLES / "baseline_did.txt", "Baseline binned DiD", baseline)

    if "high_zoning" in df.columns:
        zoning = fit(
            f"d_log_rent ~ high_inflow:post + high_inflow:post:high_zoning + low_inflow:post{rhs_controls} + C(fips) + C(year)",
            df,
        )
        write_result(TABLES / "zoning_heterogeneity.txt", "Zoning heterogeneity", zoning)
    else:
        print("No zoning file found; skipped zoning heterogeneity.")

    if "net_mig_share" in df.columns:
        continuous = fit(f"d_log_rent ~ net_mig_share{rhs_controls} + C(fips) + C(year)", df)
        write_result(TABLES / "continuous_migration.txt", "Continuous migration robustness", continuous)

    print(f"wrote model outputs to {TABLES}")


if __name__ == "__main__":
    main()
