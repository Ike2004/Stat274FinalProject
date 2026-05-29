from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "output" / "figures"
TABLES = ROOT / "output" / "tables"


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    shock = (
        df.loc[df["year"].between(2020, 2022)]
        .groupby("fips", as_index=False)
        .agg(pandemic_shock=("net_mig_share", "sum"))
        .dropna()
    )
    q25, q75 = shock["pandemic_shock"].quantile([0.25, 0.75])
    shock["migration_group"] = pd.Series(pd.NA, index=shock.index, dtype="object")
    shock.loc[shock["pandemic_shock"] >= q75, "migration_group"] = "High inflow"
    shock.loc[shock["pandemic_shock"] <= q25, "migration_group"] = "Low inflow"
    shock.loc[shock["pandemic_shock"].abs() <= 0.005, "migration_group"] = "Neutral"
    shock["high_inflow"] = (shock["migration_group"] == "High inflow").astype(int)
    out = df.merge(shock, on="fips", how="left")
    out["post"] = (out["year"] >= 2020).astype(int)
    return out


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_shock_distribution(df: pd.DataFrame) -> None:
    shock = df[["fips", "pandemic_shock"]].dropna().drop_duplicates()
    plt.figure(figsize=(7.2, 4.4))
    plt.hist(shock["pandemic_shock"], bins=35, color="#4C78A8", edgecolor="white")
    plt.axvline(shock["pandemic_shock"].quantile(0.25), color="#F58518", linestyle="--", linewidth=1.5)
    plt.axvline(shock["pandemic_shock"].quantile(0.75), color="#F58518", linestyle="--", linewidth=1.5)
    plt.title("Distribution of Pandemic Migration Shock")
    plt.xlabel("Cumulative net migration share, 2020-2022")
    plt.ylabel("Counties")
    savefig(FIGURES / "migration_shock_distribution.png")


def plot_group_trends(df: pd.DataFrame) -> None:
    sample = df[df["year"].between(2018, 2023)].dropna(subset=["d_log_rent", "migration_group"])
    avg = sample.groupby(["year", "migration_group"], as_index=False)["d_log_rent"].mean()
    order = ["High inflow", "Neutral", "Low inflow"]
    colors = {"High inflow": "#4C78A8", "Neutral": "#54A24B", "Low inflow": "#E45756"}
    plt.figure(figsize=(7.2, 4.4))
    for group in order:
        g = avg[avg["migration_group"] == group]
        plt.plot(g["year"], g["d_log_rent"], marker="o", linewidth=2, label=group, color=colors[group])
    plt.axvline(2020, color="black", linestyle=":", linewidth=1)
    plt.title("Average Rent Growth by Migration-Shock Group")
    plt.xlabel("Year")
    plt.ylabel("Mean annual log rent growth")
    plt.legend(frameon=False)
    savefig(FIGURES / "rent_growth_by_group.png")


def plot_binscatter(df: pd.DataFrame) -> None:
    sample = df[df["year"].between(2020, 2023)].dropna(subset=["net_mig_share", "d_log_rent"]).copy()
    sample["bin"] = pd.qcut(sample["net_mig_share"], q=20, duplicates="drop")
    binned = sample.groupby("bin", observed=True).agg(
        net_mig_share=("net_mig_share", "mean"),
        d_log_rent=("d_log_rent", "mean"),
    )
    fit = np.polyfit(sample["net_mig_share"], sample["d_log_rent"], deg=1)
    x = np.linspace(sample["net_mig_share"].quantile(0.01), sample["net_mig_share"].quantile(0.99), 100)
    plt.figure(figsize=(7.2, 4.4))
    plt.scatter(binned["net_mig_share"], binned["d_log_rent"], color="#4C78A8", s=35)
    plt.plot(x, fit[0] * x + fit[1], color="#E45756", linewidth=2)
    plt.title("Rent Growth and Net Migration Share")
    plt.xlabel("Net migration share")
    plt.ylabel("Annual log rent growth")
    savefig(FIGURES / "migration_binscatter.png")


def plot_event_study(df: pd.DataFrame) -> None:
    sample = df[df["year"].between(2018, 2023)].dropna(subset=["d_log_rent", "high_inflow"]).copy()
    event_terms = []
    for year in range(2018, 2024):
        if year == 2019:
            continue
        col = f"high_x_{year}"
        sample[col] = sample["high_inflow"] * (sample["year"] == year).astype(int)
        event_terms.append(col)
    res = smf.ols("d_log_rent ~ " + " + ".join(event_terms) + " + C(fips) + C(year)", data=sample).fit(
        cov_type="cluster", cov_kwds={"groups": sample["fips"]}
    )
    rows = []
    for year in range(2018, 2024):
        if year == 2019:
            rows.append({"year": year, "coef": 0.0, "se": 0.0})
            continue
        name = f"high_x_{year}"
        rows.append({"year": year, "coef": res.params[name], "se": res.bse[name]})
    event = pd.DataFrame(rows)
    event["lo"] = event["coef"] - 1.96 * event["se"]
    event["hi"] = event["coef"] + 1.96 * event["se"]
    event.to_csv(TABLES / "event_study_coefficients.csv", index=False)

    plt.figure(figsize=(7.2, 4.4))
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(2020, color="black", linestyle=":", linewidth=1)
    plt.errorbar(event["year"], event["coef"], yerr=1.96 * event["se"], fmt="o-", color="#4C78A8", capsize=3)
    plt.title("Event-Study: High-Inflow Counties Relative to 2019")
    plt.xlabel("Year")
    plt.ylabel("Coefficient on high inflow x year")
    savefig(FIGURES / "event_study_high_inflow.png")


def write_summary_table(df: pd.DataFrame) -> None:
    sample = df[df["year"].between(2018, 2023)].dropna(subset=["migration_group"])
    rows = []
    for group in ["High inflow", "Neutral", "Low inflow"]:
        g = sample[sample["migration_group"] == group]
        rows.append({
            "Group": group,
            "Counties": g["fips"].nunique(),
            "Mean rent growth": g["d_log_rent"].mean(),
            "Mean net migration share": g["net_mig_share"].mean(),
            "Mean rent": g["rent"].mean(),
        })
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "summary_by_group.csv", index=False)
    with (TABLES / "summary_by_group.tex").open("w") as f:
        f.write("\\begin{table}[!htbp]\\centering\n")
        f.write("\\caption{Summary Statistics by Migration-Shock Group}\\label{tab:summary_by_group}\n")
        f.write("\\begin{tabular}{lrrrr}\\toprule\n")
        f.write("Group & Counties & Mean rent growth & Mean net migration share & Mean rent \\\\\n")
        f.write("\\midrule\n")
        for _, r in table.iterrows():
            f.write(f"{r['Group']} & {int(r['Counties']):,} & {r['Mean rent growth']:.3f} & {r['Mean net migration share']:.3f} & {r['Mean rent']:.0f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED / "county_year_panel.csv")
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df = add_bins(df)
    plot_shock_distribution(df)
    plot_group_trends(df)
    plot_binscatter(df)
    plot_event_study(df)
    write_summary_table(df)
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
