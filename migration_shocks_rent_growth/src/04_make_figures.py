from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from importlib.machinery import SourceFileLoader


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "output" / "figures"
TABLES = ROOT / "output" / "tables"

design = SourceFileLoader("design", str(ROOT / "src" / "03_estimate_models.py")).load_module()


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(PROCESSED / "county_year_panel.csv")
    panel["fips"] = panel["fips"].astype(str).str.zfill(5)
    panel = design.add_early_pandemic_shock(panel)
    panel, _ = design.add_groups(panel)
    return panel


def plot_shock_distribution(df: pd.DataFrame) -> None:
    shock = df[["fips", "early_migration_shock"]].dropna().drop_duplicates()
    qs = shock["early_migration_shock"].quantile([0.2, 0.4, 0.6, 0.8])
    plt.figure(figsize=(7.2, 4.4))
    plt.hist(shock["early_migration_shock"], bins=35, color="#4C78A8", edgecolor="white")
    for q in qs:
        plt.axvline(q, color="#F58518", linestyle="--", linewidth=1.2)
    plt.axvline(0, color="black", linestyle=":", linewidth=1)
    plt.title("Distribution of Early-Pandemic Abnormal Migration Shock")
    plt.xlabel("2021 net migration rate minus 2019-2020 baseline")
    plt.ylabel("Counties")
    savefig(FIGURES / "early_shock_distribution.png")


def plot_group_trends(df: pd.DataFrame) -> None:
    sample = df[df["year"].between(2018, 2023)].dropna(subset=["d_log_rent", "shock_quintile"])
    avg = sample.groupby(["year", "shock_quintile"], as_index=False, observed=True)["d_log_rent"].mean()
    order = ["Q1 lowest", "Q2", "Q3 middle", "Q4", "Q5 highest"]
    colors = {
        "Q1 lowest": "#E45756",
        "Q2": "#F58518",
        "Q3 middle": "#54A24B",
        "Q4": "#72B7B2",
        "Q5 highest": "#4C78A8",
    }
    plt.figure(figsize=(7.2, 4.4))
    for group in order:
        g = avg[avg["shock_quintile"] == group]
        plt.plot(g["year"], g["d_log_rent"], marker="o", linewidth=2, label=group, color=colors[group])
    plt.axvline(2021, color="black", linestyle=":", linewidth=1)
    plt.title("Average Rent Growth by Early-Pandemic Shock Quintile")
    plt.xlabel("Year")
    plt.ylabel("Mean annual log rent growth")
    plt.legend(frameon=False, ncol=2)
    savefig(FIGURES / "rent_growth_by_early_shock_quintile.png")


def plot_event_study() -> None:
    event = pd.read_csv(TABLES / "event_study_coefficients.csv")
    plt.figure(figsize=(7.2, 4.4))
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(2021, color="black", linestyle=":", linewidth=1)
    yerr = [event["coef"] - event["lo"], event["hi"] - event["coef"]]
    plt.errorbar(event["year"], event["coef"], yerr=yerr, fmt="o-", color="#4C78A8", capsize=3)
    plt.title("Event Study: Q5 Shock Counties Relative to Q3")
    plt.xlabel("Year")
    plt.ylabel("Coefficient on high shock x year")
    savefig(FIGURES / "event_study_q5_vs_q3.png")


def plot_continuous_binscatter(df: pd.DataFrame) -> None:
    sample = df[df["year"].between(2022, 2023)].dropna(subset=["early_migration_shock", "d_log_rent"]).copy()
    sample["bin"] = pd.qcut(sample["early_migration_shock"], q=20, duplicates="drop")
    binned = sample.groupby("bin", observed=True).agg(
        early_migration_shock=("early_migration_shock", "mean"),
        d_log_rent=("d_log_rent", "mean"),
    )
    plt.figure(figsize=(7.2, 4.4))
    plt.scatter(binned["early_migration_shock"], binned["d_log_rent"], color="#4C78A8", s=35)
    plt.axvline(0, color="black", linestyle=":", linewidth=1)
    plt.title("Subsequent Rent Growth and Early-Pandemic Migration Shock")
    plt.xlabel("Early-pandemic abnormal migration shock")
    plt.ylabel("Mean annual log rent growth, 2022-2023")
    savefig(FIGURES / "shock_binscatter.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    plot_shock_distribution(df)
    plot_group_trends(df)
    plot_event_study()
    plot_continuous_binscatter(df)
    print(f"wrote revised figures to {FIGURES}")


if __name__ == "__main__":
    main()
