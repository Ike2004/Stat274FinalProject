from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"


def stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED / "county_year_panel.csv")
    df["fips"] = df["fips"].astype(str).str.zfill(5)

    shock = (
        df.loc[df["year"].between(2020, 2022)]
        .groupby("fips", as_index=False)
        .agg(pandemic_shock=("net_mig_share", "sum"))
        .dropna()
    )
    shock["shock_quintile"] = pd.qcut(shock["pandemic_shock"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    shock["quintile_label"] = shock["shock_quintile"].map({
        1: "Q1 lowest",
        2: "Q2",
        3: "Q3 middle",
        4: "Q4",
        5: "Q5 highest",
    })
    df = df.merge(shock, on="fips", how="inner")
    df["post"] = (df["year"] >= 2020).astype(int)

    for q in [1, 2, 4, 5]:
        df[f"q{q}_post"] = (df["shock_quintile"].eq(q).astype(int) * df["post"])

    model_df = df.dropna(subset=["d_log_rent", "shock_quintile"])
    res = smf.ols("d_log_rent ~ q1_post + q2_post + q4_post + q5_post + C(fips) + C(year)", data=model_df).fit(
        cov_type="cluster", cov_kwds={"groups": model_df["fips"]}
    )

    with (TABLES / "quintile_did.txt").open("w") as f:
        f.write(res.summary().as_text())

    labels = {
        "q1_post": "Q1 lowest migration $\\times$ Post",
        "q2_post": "Q2 $\\times$ Post",
        "q4_post": "Q4 $\\times$ Post",
        "q5_post": "Q5 highest migration $\\times$ Post",
    }
    with (TABLES / "quintile_did.tex").open("w") as f:
        f.write("\\begin{table}[!htbp]\\centering\n")
        f.write("\\caption{Five-Bin Migration Shock Difference-in-Differences}\\label{tab:quintile_did}\n")
        f.write("\\begin{tabular}{lc}\\toprule\n")
        f.write(" & Outcome: annual log rent growth \\\\\n")
        f.write("\\midrule\n")
        for name, label in labels.items():
            coef = res.params[name]
            se = res.bse[name]
            p = res.pvalues[name]
            f.write(f"{label} & {coef:.4f}{stars(p)} \\\\\n")
            f.write(f" & ({se:.4f}) \\\\\n")
        f.write("\\midrule\n")
        f.write("Reference group & Q3 middle migration quintile \\\\\n")
        f.write("County fixed effects & Yes \\\\\n")
        f.write("Year fixed effects & Yes \\\\\n")
        f.write(f"Observations & {int(res.nobs):,} \\\\\n")
        f.write(f"$R^2$ & {res.rsquared:.3f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\multicolumn{2}{p{0.72\\textwidth}}{\\footnotesize Notes: Counties are divided into five bins using cumulative 2020--2022 net migration share. Q3 is omitted as the reference group. Standard errors are clustered by county. Significance levels: * $p<0.10$, ** $p<0.05$, *** $p<0.01$.}\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    trends = (
        df[df["year"].between(2018, 2023)]
        .dropna(subset=["d_log_rent", "quintile_label"])
        .groupby(["year", "quintile_label"], as_index=False)["d_log_rent"]
        .mean()
    )
    plt.figure(figsize=(7.4, 4.6))
    colors = {
        "Q1 lowest": "#E45756",
        "Q2": "#F58518",
        "Q3 middle": "#54A24B",
        "Q4": "#72B7B2",
        "Q5 highest": "#4C78A8",
    }
    for label in ["Q1 lowest", "Q2", "Q3 middle", "Q4", "Q5 highest"]:
        g = trends[trends["quintile_label"] == label]
        plt.plot(g["year"], g["d_log_rent"], marker="o", linewidth=2, label=label, color=colors[label])
    plt.axvline(2020, color="black", linestyle=":", linewidth=1)
    plt.title("Average Rent Growth by Migration-Shock Quintile")
    plt.xlabel("Year")
    plt.ylabel("Mean annual log rent growth")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "rent_growth_by_quintile.png", dpi=220, bbox_inches="tight")
    plt.close()

    coef_rows = []
    for name, label in {
        "q1_post": "Q1 lowest",
        "q2_post": "Q2",
        "q4_post": "Q4",
        "q5_post": "Q5 highest",
    }.items():
        coef_rows.append({
            "label": label,
            "coef": res.params[name],
            "se": res.bse[name],
        })
    coef_df = pd.DataFrame(coef_rows)
    coef_df["lo"] = coef_df["coef"] - 1.96 * coef_df["se"]
    coef_df["hi"] = coef_df["coef"] + 1.96 * coef_df["se"]
    coef_df.to_csv(TABLES / "quintile_did_coefficients.csv", index=False)
    plt.figure(figsize=(7.0, 4.3))
    x = range(len(coef_df))
    plt.axhline(0, color="black", linewidth=1)
    plt.errorbar(x, coef_df["coef"], yerr=1.96 * coef_df["se"], fmt="o", color="#4C78A8", capsize=4)
    plt.xticks(list(x), coef_df["label"])
    plt.title("Five-Bin DiD Coefficients Relative to Q3")
    plt.xlabel("Migration-shock quintile")
    plt.ylabel("Coefficient on quintile x post")
    plt.tight_layout()
    plt.savefig(FIGURES / "quintile_did_coefficients.png", dpi=220, bbox_inches="tight")
    plt.close()

    print(f"wrote {TABLES / 'quintile_did.tex'}")


if __name__ == "__main__":
    main()
