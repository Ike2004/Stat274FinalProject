from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "output" / "tables"


def stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def add_early_pandemic_shock(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the revised early-pandemic abnormal migration shock.

    IRS year is interpreted as destination year. The shock is the 2021 net
    migration rate relative to the county's own 2019/2020 baseline.
    """
    rates = (
        panel.loc[panel["year"].isin([2019, 2020, 2021]), ["fips", "year", "net_mig_share"]]
        .dropna()
        .pivot(index="fips", columns="year", values="net_mig_share")
        .rename(columns={2019: "netmig_2019", 2020: "netmig_2020", 2021: "netmig_2021"})
        .reset_index()
    )
    needed = ["netmig_2019", "netmig_2020", "netmig_2021"]
    rates = rates.dropna(subset=needed).copy()
    rates["pre_net_mig_rate"] = 0.5 * (rates["netmig_2019"] + rates["netmig_2020"])
    rates["early_migration_shock"] = rates["netmig_2021"] - rates["pre_net_mig_rate"]

    out = panel.merge(rates, on="fips", how="left")
    return out


def add_groups(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    eligible = panel.loc[
        panel["year"].between(2018, 2023) & panel["d_log_rent"].notna() & panel["early_migration_shock"].notna(),
        ["fips", "early_migration_shock"],
    ].drop_duplicates()
    quantiles = eligible["early_migration_shock"].quantile([0.20, 0.40, 0.60, 0.80])
    p20, p40, p60, p80 = quantiles.loc[0.20], quantiles.loc[0.40], quantiles.loc[0.60], quantiles.loc[0.80]

    groups = eligible.copy()
    groups["shock_quintile"] = pd.qcut(
        groups["early_migration_shock"],
        5,
        labels=["Q1 lowest", "Q2", "Q3 middle", "Q4", "Q5 highest"],
    )
    groups["shock_group"] = pd.Series(pd.NA, index=groups.index, dtype="object")
    groups.loc[groups["early_migration_shock"] >= p80, "shock_group"] = "High shock"
    groups.loc[groups["early_migration_shock"] <= p20, "shock_group"] = "Low shock"
    groups.loc[
        groups["early_migration_shock"].between(p40, p60, inclusive="both"),
        "shock_group",
    ] = "Neutral shock"
    groups["high_shock"] = (groups["shock_group"] == "High shock").astype(int)
    groups["low_shock"] = (groups["shock_group"] == "Low shock").astype(int)
    groups["neutral_shock"] = (groups["shock_group"] == "Neutral shock").astype(int)

    out = panel.merge(groups, on=["fips", "early_migration_shock"], how="left")
    out["post"] = (out["year"] >= 2022).astype(int)
    for q in ["Q1 lowest", "Q2", "Q4", "Q5 highest"]:
        safe = q.lower().replace(" ", "_")
        out[f"{safe}_post"] = (out["shock_quintile"].eq(q).astype(int) * out["post"])

    if "permits_2020_per_capita" in out.columns:
        median_permits = out[["fips", "permits_2020_per_capita"]].drop_duplicates()["permits_2020_per_capita"].median()
        out["low_permit"] = (out["permits_2020_per_capita"] < median_permits).astype(float)

    return out, quantiles


def main_sample(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["year"].between(2018, 2023) & df["shock_quintile"].notna()].copy()


def fit(formula: str, df: pd.DataFrame):
    terms = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula))
    ignore = {"C"}
    needed = ["d_log_rent", "fips"] + sorted(t for t in terms if t in df.columns and t not in ignore)
    model_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
    res = smf.ols(formula, data=model_df).fit(cov_type="cluster", cov_kwds={"groups": model_df["fips"]})
    return res, model_df


def write_result(path: Path, title: str, res) -> None:
    with path.open("w") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        f.write(res.summary().as_text())


def write_quantiles(quantiles: pd.Series) -> None:
    out = pd.DataFrame({
        "Quantile": ["P20", "P40", "P60", "P80"],
        "Shock": [quantiles.loc[0.20], quantiles.loc[0.40], quantiles.loc[0.60], quantiles.loc[0.80]],
    })
    out.to_csv(TABLES / "shock_quantiles.csv", index=False)
    with (TABLES / "shock_quantiles.tex").open("w") as f:
        f.write("\\begin{table}[!htbp]\\centering\n")
        f.write("\\caption{Early-Pandemic Abnormal Migration Shock Cutoffs}\\label{tab:shock_quantiles}\n")
        f.write("\\begin{tabular}{lr}\\toprule\n")
        f.write("Quantile & Shock cutoff \\\\\n\\midrule\n")
        for _, row in out.iterrows():
            f.write(f"{row['Quantile']} & {row['Shock']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def write_summary(df: pd.DataFrame) -> None:
    sample = main_sample(df)
    post = sample[sample["year"].between(2022, 2023)]
    rows = []
    for group in ["Q1 lowest", "Q2", "Q3 middle", "Q4", "Q5 highest"]:
        counties = sample.loc[sample["shock_quintile"] == group, "fips"].drop_duplicates()
        g_cross = sample[sample["fips"].isin(counties)][["fips", "early_migration_shock", "netmig_2021", "permits_2020_per_capita"]].drop_duplicates("fips")
        g_post = post[post["fips"].isin(counties)]
        rows.append({
            "Group": group,
            "Counties": len(counties),
            "Mean shock": g_cross["early_migration_shock"].mean(),
            "Mean 2021 net migration rate": g_cross["netmig_2021"].mean(),
            "Mean 2022-2023 rent growth": g_post["d_log_rent"].mean(),
            "Mean 2020 permits per capita": g_cross["permits_2020_per_capita"].mean(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "shock_group_summary.csv", index=False)
    with (TABLES / "shock_group_summary.tex").open("w") as f:
        f.write("\\begin{table}[!htbp]\\centering\n")
        f.write("\\caption{Summary Statistics by Early-Pandemic Migration-Shock Quintile}\\label{tab:shock_group_summary}\n")
        f.write("\\begin{tabular}{lrrrrr}\\toprule\n")
        f.write("Group & Counties & Shock & 2021 net mig. rate & Rent growth & Permits pc \\\\\n")
        f.write("\\midrule\n")
        for _, row in out.iterrows():
            f.write(
                f"{row['Group']} & {int(row['Counties']):,} & {row['Mean shock']:.4f} & "
                f"{row['Mean 2021 net migration rate']:.4f} & {row['Mean 2022-2023 rent growth']:.4f} & "
                f"{row['Mean 2020 permits per capita']:.4f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\multicolumn{6}{p{0.88\\textwidth}}{\\footnotesize Notes: Shock is the 2021 IRS net migration rate minus the county's own 2019--2020 baseline. Rent growth is mean annual log ZORI rent growth in 2022--2023. Permits pc is 2020 residential permits per IRS population base.}\n")
        f.write("\\end{tabular}\n\\end{table}\n")


def write_reg_table(path: Path, caption: str, label: str, rows: list[tuple[str, object, str]], res, sample_note: str) -> None:
    with path.open("w") as f:
        f.write("\\begin{table}[!htbp]\\centering\n")
        f.write(f"\\caption{{{caption}}}\\label{{{label}}}\n")
        f.write("\\begin{tabular}{lc}\\toprule\n")
        f.write(" & Annual log rent growth \\\\\n\\midrule\n")
        for pretty, result, param in rows:
            coef = result.params[param]
            se = result.bse[param]
            p = result.pvalues[param]
            f.write(f"{pretty} & {coef:.4f}{stars(p)} \\\\\n")
            f.write(f" & ({se:.4f}) \\\\\n")
        f.write("\\midrule\n")
        f.write("County fixed effects & Yes \\\\\n")
        f.write("Year fixed effects & Yes \\\\\n")
        f.write(f"Observations & {int(res.nobs):,} \\\\\n")
        f.write(f"$R^2$ & {res.rsquared:.3f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write(f"\\multicolumn{{2}}{{p{{0.76\\textwidth}}}}{{\\footnotesize Notes: {sample_note} Standard errors are clustered by county. Significance levels: * $p<0.10$, ** $p<0.05$, *** $p<0.01$.}}\n")
        f.write("\\end{tabular}\n\\end{table}\n")


def estimate_event_study(df: pd.DataFrame):
    sample = df[
        df["year"].between(2018, 2023)
        & df["shock_quintile"].isin(["Q5 highest", "Q3 middle"])
    ].dropna(subset=["d_log_rent", "high_shock"]).copy()
    sample["q5_shock"] = sample["shock_quintile"].eq("Q5 highest").astype(int)
    terms = []
    for year in range(2018, 2024):
        if year == 2019:
            continue
        col = f"high_x_{year}"
        sample[col] = sample["q5_shock"] * (sample["year"] == year).astype(int)
        terms.append(col)
    formula = "d_log_rent ~ " + " + ".join(terms) + " + C(fips) + C(year)"
    res = smf.ols(formula, data=sample).fit(cov_type="cluster", cov_kwds={"groups": sample["fips"]})
    rows = []
    for year in range(2018, 2024):
        if year == 2019:
            rows.append({"year": year, "coef": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0})
        else:
            param = f"high_x_{year}"
            coef = res.params[param]
            se = res.bse[param]
            rows.append({"year": year, "coef": coef, "se": se, "lo": coef - 1.96 * se, "hi": coef + 1.96 * se})
    event = pd.DataFrame(rows)
    event.to_csv(TABLES / "event_study_coefficients.csv", index=False)
    write_result(TABLES / "event_study.txt", "Event study: Q5 shock vs Q3 middle shock", res)
    return res


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PROCESSED / "county_year_panel.csv")
    panel["fips"] = panel["fips"].astype(str).str.zfill(5)
    panel = add_early_pandemic_shock(panel)
    panel, quantiles = add_groups(panel)
    analysis = main_sample(panel)

    write_quantiles(quantiles)
    write_summary(panel)

    baseline, baseline_df = fit("d_log_rent ~ q1_lowest_post + q2_post + q4_post + q5_highest_post + C(fips) + C(year)", analysis)
    write_result(TABLES / "main_lagged_did.txt", "Main lagged five-bin DiD", baseline)
    write_reg_table(
        TABLES / "main_lagged_did.tex",
        "Lagged Five-Bin Difference-in-Differences",
        "tab:main_lagged_did",
        [
            ("Q1 lowest shock $\\times$ Post", baseline, "q1_lowest_post"),
            ("Q2 $\\times$ Post", baseline, "q2_post"),
            ("Q4 $\\times$ Post", baseline, "q4_post"),
            ("Q5 highest shock $\\times$ Post", baseline, "q5_highest_post"),
        ],
        baseline,
        "Post is 2022--2023. The reference group is Q3 middle-shock counties.",
    )

    if "low_permit" in analysis.columns:
        supply, supply_df = fit(
            "d_log_rent ~ q1_lowest_post + q2_post + q4_post + q5_highest_post + q5_highest_post:low_permit + low_permit:post + C(fips) + C(year)",
            analysis,
        )
        write_result(TABLES / "supply_heterogeneity_low_permit.txt", "Low-permit supply heterogeneity", supply)
        write_reg_table(
            TABLES / "supply_heterogeneity_low_permit.tex",
            "Supply-Constraint Heterogeneity Using Low 2020 Permits",
            "tab:supply_heterogeneity",
            [
                ("Q5 highest shock $\\times$ Post", supply, "q5_highest_post"),
                ("Q5 highest shock $\\times$ Post $\\times$ Low permit", supply, "q5_highest_post:low_permit"),
                ("Low permit $\\times$ Post", supply, "low_permit:post"),
                ("Q1 lowest shock $\\times$ Post", supply, "q1_lowest_post"),
            ],
            supply,
            "Low permit equals one for counties below the sample median of 2020 residential permits per capita.",
        )

    continuous, continuous_df = fit("d_log_rent ~ early_migration_shock:post + C(fips) + C(year)", analysis)
    write_result(TABLES / "continuous_shock.txt", "Continuous shock robustness", continuous)
    write_reg_table(
        TABLES / "continuous_shock.tex",
        "Continuous Shock Robustness",
        "tab:continuous_shock",
        [("Shock $\\times$ Post", continuous, "early_migration_shock:post")],
        continuous,
        "Shock is continuous rather than binned.",
    )

    no2023, _ = fit(
        "d_log_rent ~ q1_lowest_post + q2_post + q4_post + q5_highest_post + C(fips) + C(year)",
        analysis[analysis["year"] <= 2022],
    )
    write_result(TABLES / "main_lagged_did_exclude2023.txt", "Main lagged binned DiD excluding 2023", no2023)

    estimate_event_study(panel)
    panel.to_csv(PROCESSED / "county_year_panel_with_shock.csv", index=False)
    print(f"wrote revised model outputs to {TABLES}")


if __name__ == "__main__":
    main()
