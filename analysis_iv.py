from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
DATA = ROOT / "data"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
for p in [DATA, FIG, TAB]:
    p.mkdir(exist_ok=True)


COLORS = {
    "blue": "#2F5F8F",
    "teal": "#2A9D8F",
    "green": "#5B8E3E",
    "orange": "#D9822B",
    "red": "#B85C5C",
    "gray": "#6B7280",
}


IRS_FILES = {
    2019: "countyinflow1819.csv",
    2020: "countyinflow1920.csv",
    2021: "countyinflow2021.csv",
    2022: "countyinflow2122.csv",
}


def fips(state, county):
    return pd.Series(state).astype(int).astype(str).str.zfill(2) + pd.Series(county).astype(int).astype(str).str.zfill(3)


def setup_plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E6E8EB",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def build_zori():
    z = pd.read_csv(RAW / "zori_county.csv")
    z["fips"] = z["StateCodeFIPS"].astype(int).astype(str).str.zfill(2) + z[
        "MunicipalCodeFIPS"
    ].astype(int).astype(str).str.zfill(3)
    id_cols = ["fips", "RegionName", "StateName", "Metro"]
    date_cols = [c for c in z.columns if c[:4].isdigit()]
    long = z[id_cols + date_cols].melt(id_vars=id_cols, var_name="date", value_name="rent")
    long["date"] = pd.to_datetime(long["date"])
    long["year"] = long["date"].dt.year
    annual = (
        long.dropna(subset=["rent"])
        .groupby(["fips", "RegionName", "StateName", "Metro", "year"], as_index=False)
        .agg(rent=("rent", "mean"), months=("rent", "size"))
    )
    annual = annual[annual["months"] >= 9].copy()
    annual = annual.sort_values(["fips", "year"])
    annual["log_rent"] = np.log(annual["rent"])
    annual["rent_growth"] = annual.groupby("fips")["log_rent"].diff()
    return annual


def build_irs_migration():
    rows = []
    for year, fn in IRS_FILES.items():
        x = pd.read_csv(RAW / fn, encoding="latin1")
        x["fips"] = fips(x["y2_statefips"], x["y2_countyfips"])
        x["origin_fips"] = fips(x["y1_statefips"], x["y1_countyfips"])
        x["n2"] = pd.to_numeric(x["n2"], errors="coerce").fillna(0)
        domestic = x[(x["y1_statefips"].astype(int) == 97) & (x["y1_countyfips"].astype(int) == 0)][
            ["fips", "n2"]
        ].rename(columns={"n2": "domestic_inflow"})
        nonmig = x[x["origin_fips"] == x["fips"]][["fips", "n2"]].rename(columns={"n2": "nonmigrant"})
        out = domestic.merge(nonmig, on="fips", how="left")
        out["irs_base_people"] = out["domestic_inflow"].fillna(0) + out["nonmigrant"].fillna(0)
        out["irs_inflow_rate"] = out["domestic_inflow"] / out["irs_base_people"].replace(0, np.nan)
        out["year"] = year
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def build_population():
    p20 = pd.read_csv(RAW / "co-est2020-alldata.csv", encoding="latin1")
    p23 = pd.read_csv(RAW / "co-est2023-alldata.csv", encoding="latin1")
    p20 = p20[p20["SUMLEV"] == 50].copy()
    p23 = p23[p23["SUMLEV"] == 50].copy()
    p20["fips"] = fips(p20["STATE"], p20["COUNTY"])
    p23["fips"] = fips(p23["STATE"], p23["COUNTY"])
    pop = p20[["fips", "STNAME", "CTYNAME", "POPESTIMATE2018", "POPESTIMATE2019", "POPESTIMATE2020"]].merge(
        p23[["fips", "POPESTIMATE2021", "POPESTIMATE2022"]], on="fips", how="inner"
    )
    rows = []
    for year in [2019, 2020, 2021, 2022]:
        d = pop[["fips", "STNAME", "CTYNAME", f"POPESTIMATE{year - 1}", f"POPESTIMATE{year}"]].copy()
        d = d.rename(columns={f"POPESTIMATE{year - 1}": "pop_lag", f"POPESTIMATE{year}": "population"})
        d["year"] = year
        d["pop_growth"] = np.log(d["population"]) - np.log(d["pop_lag"])
        rows.append(d[["fips", "STNAME", "CTYNAME", "year", "population", "pop_growth"]])
    return pd.concat(rows, ignore_index=True)


def build_permits():
    b = pd.read_csv(RAW / "bps_2020_county.txt", skiprows=3, header=None)
    b["fips"] = fips(b.iloc[:, 1], b.iloc[:, 2])
    unit_cols = [7, 10, 13, 16]
    for c in unit_cols:
        b.iloc[:, c] = pd.to_numeric(b.iloc[:, c], errors="coerce").fillna(0)
    out = b[["fips"]].copy()
    out["permits_2020"] = b.iloc[:, unit_cols].sum(axis=1)
    return out


def residualize(y, x):
    x = sm.add_constant(x, has_constant="add")
    return sm.OLS(y, x, missing="drop").fit().resid


def iv_one_endog(y, d, z, x=None):
    data = pd.DataFrame({"y": y, "d": d, "z": z}).reset_index(drop=True)
    if x is not None and len(x.columns) > 0:
        xdf = pd.DataFrame(x).reset_index(drop=True)
        xdf.columns = [f"x{i}" for i in range(xdf.shape[1])]
        data = pd.concat([data, xdf], axis=1).dropna()
        xcols = xdf.columns.tolist()
        yr = residualize(data["y"], data[xcols])
        dr = residualize(data["d"], data[xcols])
        zr = residualize(data["z"], data[xcols])
    else:
        data = data.dropna()
        yr = data["y"] - data["y"].mean()
        dr = data["d"] - data["d"].mean()
        zr = data["z"] - data["z"].mean()

    beta = float(np.sum(zr * yr) / np.sum(zr * dr))
    first = sm.OLS(dr, sm.add_constant(zr)).fit(cov_type="HC1")
    reduced = sm.OLS(yr, sm.add_constant(zr)).fit(cov_type="HC1")

    u = yr - beta * dr
    denom = float(np.sum(zr * dr))
    meat = float(np.sum((zr * u) ** 2))
    n = len(data)
    se = np.sqrt(meat / (denom**2)) * np.sqrt(n / max(n - 2, 1))

    return {
        "estimate": beta,
        "std_error": se,
        "ci_low": beta - 1.96 * se,
        "ci_high": beta + 1.96 * se,
        "first_stage_coef": float(first.params.iloc[1]),
        "first_stage_se": float(first.bse.iloc[1]),
        "first_stage_f": float(first.tvalues.iloc[1] ** 2),
        "reduced_form_coef": float(reduced.params.iloc[1]),
        "reduced_form_se": float(reduced.bse.iloc[1]),
        "n": int(n),
    }


def main():
    setup_plot_style()
    rent = build_zori()
    migration = build_irs_migration()
    pop = build_population()
    permits = build_permits()

    panel = rent.merge(migration, on=["fips", "year"], how="inner")
    panel = panel.merge(pop, on=["fips", "year"], how="inner")
    panel = panel.merge(permits, on="fips", how="left")
    panel["permit_rate_2020"] = panel["permits_2020"] / panel["population"].replace(0, np.nan) * 1000
    panel["state"] = panel["fips"].str[:2]
    panel = panel.dropna(subset=["rent_growth", "irs_inflow_rate", "pop_growth", "rent", "permit_rate_2020"])
    panel.to_csv(DATA / "iv_population_panel.csv", index=False)

    wide = panel.pivot_table(
        index=["fips", "RegionName", "StateName", "state"],
        columns="year",
        values=["rent_growth", "irs_inflow_rate", "pop_growth", "rent", "permit_rate_2020"],
        observed=False,
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["rent_acceleration"] = wide[["rent_growth_2021", "rent_growth_2022"]].mean(axis=1) - wide[
        ["rent_growth_2019", "rent_growth_2020"]
    ].mean(axis=1)
    wide["population_growth_shock"] = wide[["pop_growth_2021", "pop_growth_2022"]].mean(axis=1) - wide[
        ["pop_growth_2019", "pop_growth_2020"]
    ].mean(axis=1)
    wide["migration_shock_iv"] = wide[["irs_inflow_rate_2021", "irs_inflow_rate_2022"]].mean(axis=1) - wide[
        ["irs_inflow_rate_2019", "irs_inflow_rate_2020"]
    ].mean(axis=1)
    wide["pre_growth"] = wide[["rent_growth_2019", "rent_growth_2020"]].mean(axis=1)
    wide["log_rent_2020"] = np.log(wide["rent_2020"])
    wide = wide.dropna(
        subset=[
            "rent_acceleration",
            "population_growth_shock",
            "migration_shock_iv",
            "pre_growth",
            "log_rent_2020",
            "permit_rate_2020_2020",
        ]
    )
    wide["shock_bin"] = pd.qcut(wide["migration_shock_iv"], 3, labels=["Low shock", "Medium shock", "High shock"])
    wide.to_csv(DATA / "iv_population_county_shock.csv", index=False)

    base_controls = wide[["pre_growth", "log_rent_2020", "permit_rate_2020_2020"]]
    state_controls = pd.concat(
        [base_controls, pd.get_dummies(wide["state"], prefix="state", drop_first=True, dtype=float)], axis=1
    )
    specs = [
        ("No controls", None),
        ("Baseline controls", base_controls),
        ("Baseline controls + state FE", state_controls),
    ]
    results = []
    for name, controls in specs:
        res = iv_one_endog(
            y=wide["rent_acceleration"],
            d=wide["population_growth_shock"],
            z=wide["migration_shock_iv"],
            x=controls,
        )
        res["specification"] = name
        results.append(res)
    results = pd.DataFrame(results)
    results.to_csv(TAB / "iv_population_results.csv", index=False)
    results[results["specification"] == "Baseline controls"].to_csv(TAB / "iv_population_main.csv", index=False)

    wide["low_permit"] = (
        wide["permit_rate_2020_2020"] <= wide["permit_rate_2020_2020"].median()
    ).astype(int)
    heterogeneity_rows = []
    for group_value, group_name in [(1, "Low-permit / more constrained"), (0, "High-permit / less constrained")]:
        g = wide[wide["low_permit"] == group_value].copy()
        g_controls = g[["pre_growth", "log_rent_2020"]]
        h = iv_one_endog(
            y=g["rent_acceleration"],
            d=g["population_growth_shock"],
            z=g["migration_shock_iv"],
            x=g_controls,
        )
        h["group"] = group_name
        h["counties"] = len(g)
        heterogeneity_rows.append(h)
    pd.DataFrame(heterogeneity_rows).to_csv(TAB / "iv_population_supply_heterogeneity.csv", index=False)

    fs_rf = pd.DataFrame(
        [
            {
                "model": "First stage",
                "dependent_variable": "Population-growth shock",
                "estimate": results.loc[results["specification"] == "Baseline controls", "first_stage_coef"].iloc[0],
                "std_error": results.loc[results["specification"] == "Baseline controls", "first_stage_se"].iloc[0],
                "f_stat": results.loc[results["specification"] == "Baseline controls", "first_stage_f"].iloc[0],
            },
            {
                "model": "Reduced form",
                "dependent_variable": "Rent-growth acceleration",
                "estimate": results.loc[results["specification"] == "Baseline controls", "reduced_form_coef"].iloc[0],
                "std_error": results.loc[results["specification"] == "Baseline controls", "reduced_form_se"].iloc[0],
                "f_stat": np.nan,
            },
        ]
    )
    fs_rf.to_csv(TAB / "iv_population_first_stage_reduced_form.csv", index=False)

    desc = wide[
        ["rent_acceleration", "population_growth_shock", "migration_shock_iv", "pre_growth", "permit_rate_2020_2020"]
    ].agg(["mean", "std", "min", "max"]).T
    desc.to_csv(TAB / "iv_population_descriptive_stats.csv")

    summary = (
        wide.groupby("shock_bin", observed=False)
        .agg(
            counties=("fips", "nunique"),
            migration_shock=("migration_shock_iv", "mean"),
            population_growth_shock=("population_growth_shock", "mean"),
            rent_acceleration=("rent_acceleration", "mean"),
            pre_growth=("pre_growth", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(TAB / "iv_population_summary_by_shock_bin.csv", index=False)

    top = wide.sort_values("migration_shock_iv", ascending=False).head(10)[
        ["RegionName", "StateName", "migration_shock_iv", "population_growth_shock", "rent_acceleration"]
    ]
    top.to_csv(TAB / "iv_population_top_shock_counties.csv", index=False)

    plot_panel = panel.merge(wide[["fips", "shock_bin"]], on="fips", how="inner")
    event = plot_panel.groupby(["year", "shock_bin"], observed=False)["rent_growth"].mean().reset_index()
    plt.figure(figsize=(6.6, 4.1))
    for label, color in [("Low shock", COLORS["blue"]), ("Medium shock", COLORS["orange"]), ("High shock", COLORS["green"])]:
        d = event[event["shock_bin"].astype(str) == label]
        plt.plot(d["year"], 100 * d["rent_growth"], marker="o", markersize=5, linewidth=2.1, label=label, color=color)
    plt.axvspan(2020.5, 2022.5, color=COLORS["gray"], alpha=0.12, label="Migration-shock years")
    plt.ylabel("Annual log rent growth (%)")
    plt.xlabel("IRS migration destination year")
    plt.xticks([2019, 2020, 2021, 2022])
    plt.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG / "iv_population_shock_timeline.pdf")
    plt.savefig(FIG / "iv_population_shock_timeline.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6.1, 4.0))
    plt.scatter(
        100 * wide["migration_shock_iv"],
        100 * wide["population_growth_shock"],
        s=22,
        alpha=0.52,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.3,
    )
    x = 100 * wide["migration_shock_iv"].to_numpy()
    y = 100 * wide["population_growth_shock"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    plt.plot(xx, intercept + slope * xx, color=COLORS["red"], linewidth=2.0)
    plt.xlabel("IRS migration shock (pp)")
    plt.ylabel("Population-growth shock (pp)")
    plt.tight_layout()
    plt.savefig(FIG / "iv_population_first_stage_scatter.pdf")
    plt.savefig(FIG / "iv_population_first_stage_scatter.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6.1, 4.0))
    plt.scatter(
        100 * wide["migration_shock_iv"],
        100 * wide["rent_acceleration"],
        s=22,
        alpha=0.52,
        color=COLORS["teal"],
        edgecolor="white",
        linewidth=0.3,
    )
    x = 100 * wide["migration_shock_iv"].to_numpy()
    y = 100 * wide["rent_acceleration"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    plt.plot(xx, intercept + slope * xx, color=COLORS["red"], linewidth=2.0)
    plt.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
    plt.xlabel("IRS migration shock (pp)")
    plt.ylabel("Rent-growth acceleration (pp)")
    plt.tight_layout()
    plt.savefig(FIG / "iv_population_reduced_form_scatter.pdf")
    plt.savefig(FIG / "iv_population_reduced_form_scatter.png", dpi=200)
    plt.close()

    coef = results.copy()
    coef["effect_1pp"] = coef["estimate"]
    coef["low_1pp"] = coef["ci_low"]
    coef["high_1pp"] = coef["ci_high"]
    plt.figure(figsize=(6.2, 3.7))
    y = np.arange(len(coef))
    plt.errorbar(
        coef["effect_1pp"],
        y,
        xerr=[coef["effect_1pp"] - coef["low_1pp"], coef["high_1pp"] - coef["effect_1pp"]],
        fmt="o",
        capsize=4,
        color=COLORS["blue"],
        ecolor=COLORS["gray"],
        elinewidth=1.8,
    )
    plt.axvline(0, color="#333333", linewidth=0.8)
    plt.yticks(y, coef["specification"])
    plt.xlabel("Effect of 1 pp population-growth shock on rent acceleration (pp)")
    plt.tight_layout()
    plt.savefig(FIG / "iv_population_coefficient_plot.pdf")
    plt.savefig(FIG / "iv_population_coefficient_plot.png", dpi=200)
    plt.close()

    print("IV sample counties", len(wide))
    print(results[["specification", "estimate", "std_error", "ci_low", "ci_high", "first_stage_f", "n"]])


if __name__ == "__main__":
    main()
