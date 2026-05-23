import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)

ZORI_URL = "https://files.zillowstatic.com/research/public_csvs/zori/County_zori_uc_sfrcondomfr_sm_sa_month.csv"
POP_2019_URL = "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv"
POP_2023_URL = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv"
BPS_URL = "https://www2.census.gov/econ/bps/County/co{yy}12y.txt"
ACS_VARS = {
    "population": "B01003_001E",
    "median_income": "B19013_001E",
    "median_gross_rent": "B25064_001E",
    "housing_units": "B25001_001E",
    "occupied_units": "B25002_002E",
    "vacant_units": "B25002_003E",
    "renter_units": "B25003_003E",
    "labor_force": "B23025_003E",
    "unemployed": "B23025_005E",
}


def download_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def fetch_acs_snapshot() -> pd.DataFrame:
    """County ACS controls from Census Reporter's keyless public API.

    The Census Bureau API currently requires a key in this environment, so the
    analysis uses Census Reporter as a mirror for the latest ACS 5-year county
    controls. These variables are time-invariant controls in the panel.
    """
    cache = DATA / "censusreporter_latest_county_controls.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"fips": str})

    table_ids = "B01003,B19013,B25064,B25001,B25002,B25003,B23025"
    response = requests.get(
        "https://api.censusreporter.org/1.0/data/show/latest",
        params={"table_ids": table_ids, "geo_ids": "050|01000US"},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()["data"]
    rows = []
    for geoid, tables in payload.items():
        fips = geoid.replace("05000US", "")
        get = lambda table, cell: tables.get(table, {}).get("estimate", {}).get(cell, np.nan)
        labor_force = get("B23025", "B23025003")
        unemployed = get("B23025", "B23025005")
        housing_units = get("B25001", "B25001001")
        occupied_units = get("B25002", "B25002002")
        vacant_units = get("B25002", "B25002003")
        renter_units = get("B25003", "B25003003")
        rows.append(
            {
                "fips": fips,
                "acs_population": get("B01003", "B01003001"),
                "median_income": get("B19013", "B19013001"),
                "median_gross_rent": get("B25064", "B25064001"),
                "housing_units": housing_units,
                "vacancy_rate": vacant_units / housing_units if housing_units else np.nan,
                "renter_share": renter_units / occupied_units if occupied_units else np.nan,
                "unemployment_rate": unemployed / labor_force if labor_force else np.nan,
            }
        )
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    df.to_csv(cache, index=False)
    return df


def make_population_panel() -> pd.DataFrame:
    path_2019 = DATA / "co-est2019-alldata.csv"
    path_2023 = DATA / "co-est2023-alldata.csv"
    download_file(POP_2019_URL, path_2019)
    download_file(POP_2023_URL, path_2023)
    p19 = pd.read_csv(path_2019, encoding="latin1")
    p23 = pd.read_csv(path_2023, encoding="latin1")
    p19 = p19[p19["SUMLEV"] == 50].copy()
    p23 = p23[p23["SUMLEV"] == 50].copy()

    rows = []
    for df, years in [(p19, [2018, 2019]), (p23, [2020, 2021, 2022, 2023])]:
        for year in years:
            pop_col = f"POPESTIMATE{year}"
            net_col = f"NETMIG{year}"
            dom_col = f"DOMESTICMIG{year}"
            frame = df[["STATE", "COUNTY", "STNAME", "CTYNAME", pop_col, net_col, dom_col]].copy()
            frame["year"] = year
            frame["fips"] = frame["STATE"].astype(str).str.zfill(2) + frame["COUNTY"].astype(str).str.zfill(3)
            frame = frame.rename(columns={pop_col: "population", net_col: "net_migration", dom_col: "domestic_migration"})
            rows.append(frame[["fips", "year", "population", "net_migration", "domestic_migration"]])
    pop = pd.concat(rows, ignore_index=True).sort_values(["fips", "year"])
    pop["lag_population"] = pop.groupby("fips")["population"].shift(1)
    pop["pop_growth"] = pop.groupby("fips")["population"].transform(lambda s: np.log(s).diff())
    pop["net_migration_rate"] = pop["net_migration"] / pop["lag_population"]
    pop["domestic_migration_rate"] = pop["domestic_migration"] / pop["lag_population"]
    return pop


def make_permits_panel() -> pd.DataFrame:
    rows = []
    for year in range(2018, 2024):
        yy = str(year)[2:]
        path = DATA / f"bps_county_{year}.txt"
        download_file(BPS_URL.format(yy=yy), path)
        df = pd.read_csv(path, skiprows=3, header=None, dtype={1: str, 2: str})
        df = df.dropna(subset=[1, 2])
        df["fips"] = df[1].str.zfill(2) + df[2].str.zfill(3)
        # Total housing units authorized by permits across 1-unit, 2-unit,
        # 3-4 unit, and 5+ unit structures.
        for col in [7, 10, 13, 16]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["permits"] = df[[7, 10, 13, 16]].sum(axis=1)
        df["year"] = year
        rows.append(df[["fips", "year", "permits"]])
    permits = pd.concat(rows, ignore_index=True)
    permits = permits.groupby(["fips", "year"], as_index=False)["permits"].sum()
    permits.to_csv(DATA / "bps_county_annual_permits_2018_2023.csv", index=False)
    return permits


def make_zori_panel() -> pd.DataFrame:
    zori_path = DATA / "zori_county_smoothed_sa_month.csv"
    download_file(ZORI_URL, zori_path)
    zori = pd.read_csv(zori_path)
    date_cols = [c for c in zori.columns if c[:4].isdigit()]
    id_cols = [c for c in zori.columns if c not in date_cols]
    long = zori.melt(id_vars=id_cols, value_vars=date_cols, var_name="date", value_name="zori")
    long["date"] = pd.to_datetime(long["date"])
    long["year"] = long["date"].dt.year
    long["month"] = long["date"].dt.month
    annual = (
        long.dropna(subset=["zori"])
        .groupby(["RegionID", "RegionName", "StateName", "StateCodeFIPS", "MunicipalCodeFIPS", "year"], as_index=False)
        .agg(zori=("zori", "mean"), months=("zori", "size"))
    )
    annual = annual[annual["months"] >= 6].copy()
    annual["fips"] = annual["StateCodeFIPS"].astype(int).astype(str).str.zfill(2) + annual[
        "MunicipalCodeFIPS"
    ].astype(int).astype(str).str.zfill(3)
    annual = annual.sort_values(["fips", "year"])
    annual["rent_growth"] = annual.groupby("fips")["zori"].transform(lambda s: np.log(s).diff())
    annual["lag_zori"] = annual.groupby("fips")["zori"].shift(1)
    return annual


def build_panel() -> pd.DataFrame:
    zori = make_zori_panel()
    pop = make_population_panel()
    permits = make_permits_panel()
    acs = fetch_acs_snapshot()
    panel = zori.merge(pop, on=["fips", "year"], how="inner").merge(permits, on=["fips", "year"], how="left").merge(acs, on="fips", how="left")
    panel = panel[(panel["year"] >= 2019) & (panel["year"] <= 2023)].copy()
    panel["permits"] = panel["permits"].fillna(0)
    panel["permits_per_1k"] = 1000.0 * panel["permits"] / panel["population"]
    pre_permits = (
        permits.merge(pop[["fips", "year", "population"]], on=["fips", "year"], how="left")
        .assign(permits_per_1k=lambda x: 1000.0 * x["permits"] / x["population"])
    )
    pre_permits = (
        pre_permits[pre_permits["year"].isin([2018, 2019])]
        .groupby("fips", as_index=False)
        .agg(pre_permits_per_1k=("permits_per_1k", "mean"))
    )
    panel = panel.merge(pre_permits, on="fips", how="left")
    panel["pre_permits_per_1k"] = panel["pre_permits_per_1k"].fillna(0)

    def zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std(ddof=0)

    panel["supply_constraint"] = (
        -zscore(panel["vacancy_rate"])
        + zscore(panel["renter_share"])
        + zscore(panel["median_gross_rent"])
        - zscore(panel["pre_permits_per_1k"])
    ) / 4.0
    panel["post2020"] = (panel["year"] >= 2020).astype(int)
    panel["pop_growth_pp"] = panel["pop_growth"] * 100.0
    panel["net_migration_rate_pp"] = panel["net_migration_rate"] * 100.0
    panel["rent_growth_pct"] = panel["rent_growth"] * 100.0
    needed = [
        "rent_growth",
        "net_migration_rate",
        "median_income",
        "vacancy_rate",
        "renter_share",
        "unemployment_rate",
        "lag_zori",
        "pre_permits_per_1k",
        "supply_constraint",
    ]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    panel.to_csv(RESULTS / "county_year_panel.csv", index=False)
    return panel


def demean_by_groups(df: pd.DataFrame, cols: list[str], groups: list[str], iterations: int = 8) -> pd.DataFrame:
    out = df[cols].astype(float).copy()
    for _ in range(iterations):
        for group in groups:
            out = out - out.groupby(df[group]).transform("mean")
    return out


def cluster_se(x: np.ndarray, residuals: np.ndarray, clusters: pd.Series) -> np.ndarray:
    xtx_inv = np.linalg.inv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for _, idx in pd.Series(np.arange(len(clusters))).groupby(clusters).groups.items():
        xg = x[list(idx), :]
        ug = residuals[list(idx)]
        score = xg.T @ ug
        meat += np.outer(score, score)
    g = clusters.nunique()
    n, k = x.shape
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    return np.sqrt(np.diag(correction * xtx_inv @ meat @ xtx_inv))


def ols_fe(df: pd.DataFrame, controls: list[str], interaction: bool = False) -> dict:
    work = df.copy()
    treatment = "net_migration_rate"
    variables = ["rent_growth", treatment] + controls
    if interaction:
        work["pop_x_constraint"] = work[treatment] * work["supply_constraint"]
        variables += ["pop_x_constraint", "supply_constraint"]
    dm = demean_by_groups(work, variables, ["fips", "year"])
    y = dm["rent_growth"].to_numpy()
    xcols = [treatment] + controls
    if interaction:
        xcols = [treatment, "pop_x_constraint"] + controls
    x = dm[xcols].to_numpy()
    model = sm.OLS(y, x).fit()
    se = cluster_se(x, model.resid, work["fips"])
    return {
        "coef": model.params,
        "se": se,
        "xcols": xcols,
        "n": int(len(work)),
        "counties": int(work["fips"].nunique()),
        "r2": float(model.rsquared),
    }


def dml_plr(df: pd.DataFrame, trim: bool = False, learner_name: str = "rf") -> dict:
    controls = [
        "median_income",
        "vacancy_rate",
        "renter_share",
        "unemployment_rate",
        "lag_zori",
        "supply_constraint",
        "median_gross_rent",
        "pre_permits_per_1k",
        "population",
        "pop_growth",
    ]
    work = df.copy()
    fe_dummies = pd.get_dummies(work["year"].astype(str), prefix="year", drop_first=True, dtype=float)
    x = pd.concat([work[controls].reset_index(drop=True), fe_dummies.reset_index(drop=True)], axis=1)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(x.median(numeric_only=True))
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    y = work["rent_growth"].to_numpy()
    d = work["net_migration_rate"].to_numpy()

    y_res = np.zeros(len(work))
    d_res = np.zeros(len(work))
    kf = KFold(n_splits=5, shuffle=True, random_state=274)
    for train, test in kf.split(xs):
        if learner_name == "gb":
            y_model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, random_state=274)
            d_model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, random_state=275)
        elif learner_name == "lasso":
            y_model = LassoCV(cv=5, random_state=274, max_iter=10000)
            d_model = LassoCV(cv=5, random_state=275, max_iter=10000)
        else:
            y_model = RandomForestRegressor(n_estimators=300, min_samples_leaf=8, random_state=274, n_jobs=-1)
            d_model = RandomForestRegressor(n_estimators=300, min_samples_leaf=8, random_state=275, n_jobs=-1)
        y_model.fit(xs[train], y[train])
        d_model.fit(xs[train], d[train])
        y_res[test] = y[test] - y_model.predict(xs[test])
        d_res[test] = d[test] - d_model.predict(xs[test])

    keep = np.ones(len(work), dtype=bool)
    if trim:
        lo, hi = np.quantile(d_res, [0.02, 0.98])
        keep = (d_res >= lo) & (d_res <= hi)
    x_ols = d_res[keep][:, None]
    fit = sm.OLS(y_res[keep], x_ols).fit()
    se = cluster_se(x_ols, fit.resid, work.loc[keep, "fips"])
    theta = float(fit.params[0])
    return {
        "coef": theta,
        "se": float(se[0]),
        "n": int(keep.sum()),
        "counties": int(work.loc[keep, "fips"].nunique()),
        "y_rmse": float(math.sqrt(mean_squared_error(y, y - y_res))),
        "d_r2": float(r2_score(d, d - d_res)),
    }


def fmt_ci(coef: float, se: float, scale: float = 1.0) -> str:
    lo = (coef - 1.96 * se) * scale
    hi = (coef + 1.96 * se) * scale
    return f"[{lo:.2f}, {hi:.2f}]"


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_figures(panel: pd.DataFrame, results: dict) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    blue = "#2f5d8c"
    red = "#b84a3a"
    green = "#3f7f5f"
    gray = "#666666"

    yearly = panel.groupby("year", as_index=False).agg(
        rent_growth=("rent_growth_pct", "mean"),
        migration=("net_migration_rate_pp", "mean"),
        zori=("zori", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
    ax1.plot(yearly["year"], yearly["rent_growth"], marker="o", color=blue, linewidth=2.2, label="Rent growth")
    ax1.set_ylabel("Annual rent growth, percent points", color=blue)
    ax1.tick_params(axis="y", labelcolor=blue)
    ax1.set_xlabel("Year")
    ax1.set_xticks(yearly["year"])
    ax2 = ax1.twinx()
    ax2.plot(yearly["year"], yearly["migration"], marker="s", color=red, linewidth=2.2, label="Net migration")
    ax2.set_ylabel("Net migration rate, percent points", color=red)
    ax2.tick_params(axis="y", labelcolor=red)
    ax1.set_title("Average Rent Growth and Net Migration in the Estimation Sample")
    save_fig(fig, "trend_rent_migration.png")

    plot = panel.copy()
    plot["mig_bin"] = pd.qcut(plot["net_migration_rate_pp"], 25, duplicates="drop")
    binned = plot.groupby("mig_bin", observed=True).agg(
        mig=("net_migration_rate_pp", "mean"),
        rent=("rent_growth_pct", "mean"),
        n=("rent_growth_pct", "size"),
    ).reset_index(drop=True)
    slope, intercept = np.polyfit(plot["net_migration_rate_pp"], plot["rent_growth_pct"], 1)
    xs = np.linspace(binned["mig"].min(), binned["mig"].max(), 100)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(binned["mig"], binned["rent"], s=np.sqrt(binned["n"]) * 9, color=blue, alpha=0.8, edgecolor="white")
    ax.plot(xs, intercept + slope * xs, color=red, linewidth=2.0)
    ax.axhline(0, color=gray, linewidth=0.8)
    ax.axvline(0, color=gray, linewidth=0.8)
    ax.set_xlabel("Net migration rate, percent points")
    ax.set_ylabel("Annual rent growth, percent points")
    ax.set_title("Binned Relationship Between Migration and Rent Growth")
    save_fig(fig, "binned_scatter_migration_rent.png")

    estimates = [
        ("FE", results["fe"]["coef"][0], results["fe"]["se"][0]),
        ("FE + controls", results["fe_controls"]["coef"][0], results["fe_controls"]["se"][0]),
        ("DML RF", results["dml_rf"]["coef"], results["dml_rf"]["se"]),
        ("DML RF trimmed", results["dml_trim"]["coef"], results["dml_trim"]["se"]),
        ("DML lasso", results["dml_lasso"]["coef"], results["dml_lasso"]["se"]),
        ("DML boosting", results["dml_gb"]["coef"], results["dml_gb"]["se"]),
    ]
    labels = [x[0] for x in estimates]
    coefs = np.array([x[1] for x in estimates])
    ses = np.array([x[2] for x in estimates])
    ypos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.errorbar(coefs, ypos, xerr=1.96 * ses, fmt="o", color=blue, ecolor=gray, elinewidth=1.8, capsize=4)
    ax.axvline(0, color=red, linewidth=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Effect on rent growth from 1 p.p. higher net migration")
    ax.set_title("Main Estimates with 95 Percent Confidence Intervals")
    save_fig(fig, "coefficient_plot.png")

    hetero = panel.copy()
    hetero["migration_quartile"] = pd.qcut(hetero["net_migration_rate_pp"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    hetero["constraint_group"] = np.where(
        hetero["supply_constraint"] >= hetero["supply_constraint"].median(),
        "More constrained",
        "Less constrained",
    )
    grouped = hetero.groupby(["migration_quartile", "constraint_group"], observed=True)["rent_growth_pct"].mean().reset_index()
    pivot = grouped.pivot(index="migration_quartile", columns="constraint_group", values="rent_growth_pct")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(pivot.index))
    width = 0.36
    ax.bar(x - width / 2, pivot["Less constrained"], width, color=green, label="Less constrained")
    ax.bar(x + width / 2, pivot["More constrained"], width, color=red, label="More constrained")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_xlabel("Net migration-rate quartile")
    ax.set_ylabel("Mean annual rent growth, percent points")
    ax.set_title("Rent Growth by Migration Pressure and Supply Constraint")
    ax.legend(frameon=False)
    save_fig(fig, "heterogeneity_quartiles.png")


def write_tex(results: dict, panel: pd.DataFrame) -> None:
    desc = panel[["rent_growth_pct", "net_migration_rate_pp", "zori", "population", "median_income", "vacancy_rate", "renter_share", "pre_permits_per_1k"]].describe(
        percentiles=[0.25, 0.5, 0.75]
    )
    desc_rows = []
    labels = {
        "rent_growth_pct": "Annual rent growth (pct. points)",
        "net_migration_rate_pp": "Net migration rate (pct. points)",
        "zori": "Mean annual ZORI rent (dollars)",
        "population": "Census population estimate",
        "median_income": "Median household income",
        "vacancy_rate": "Vacancy rate",
        "renter_share": "Renter share",
        "pre_permits_per_1k": "Pre-pandemic permits per 1,000 residents",
    }
    for var, label in labels.items():
        row = desc[var]
        desc_rows.append(f"{label} & {row['mean']:.2f} & {row['std']:.2f} & {row['25%']:.2f} & {row['50%']:.2f} & {row['75%']:.2f} \\\\")

    fe = results["fe"]
    fec = results["fe_controls"]
    dml = results["dml_rf"]
    dml_trim = results["dml_trim"]
    het = results["heterogeneity"]
    q25, q75 = panel["net_migration_rate"].quantile([0.25, 0.75])
    iqr_effect = dml["coef"] * (q75 - q25) * 100.0
    lines = [
        "% Auto-generated by analysis.py",
        f"\\newcommand{{\\AnalysisN}}{{{len(panel):,}}}",
        f"\\newcommand{{\\AnalysisCounties}}{{{panel['fips'].nunique():,}}}",
        f"\\newcommand{{\\AnalysisYears}}{{{int(panel['year'].min())}--{int(panel['year'].max())}}}",
        f"\\newcommand{{\\DMLIqrEffect}}{{{iqr_effect:.2f}}}",
        f"\\newcommand{{\\PopGrowthIQR}}{{{(q75 - q25) * 100:.2f}}}",
        "\\newcommand{\\DescStatsRows}{%",
        *desc_rows,
        "}",
        "\\newcommand{\\MainResultsRows}{%",
        f"Net migration rate & {fe['coef'][0]:.2f} & {fec['coef'][0]:.2f} & {dml['coef']:.2f} & {dml_trim['coef']:.2f} \\\\",
        f"95\\% CI & {fmt_ci(fe['coef'][0], fe['se'][0])} & {fmt_ci(fec['coef'][0], fec['se'][0])} & {fmt_ci(dml['coef'], dml['se'])} & {fmt_ci(dml_trim['coef'], dml_trim['se'])} \\\\",
        f"Observations & {fe['n']:,} & {fec['n']:,} & {dml['n']:,} & {dml_trim['n']:,} \\\\",
        f"Counties & {fe['counties']:,} & {fec['counties']:,} & {dml['counties']:,} & {dml_trim['counties']:,} \\\\",
        "}",
        "\\newcommand{\\HeterogeneityRows}{%",
        f"Net migration rate & {het['coef'][0]:.2f} & {fmt_ci(het['coef'][0], het['se'][0])} \\\\",
        f"Net migration rate $\\times$ supply constraint & {het['coef'][1]:.2f} & {fmt_ci(het['coef'][1], het['se'][1])} \\\\",
        f"Observations & {het['n']:,} &  \\\\",
        "}",
        f"\\newcommand{{\\OutcomeRMSE}}{{{dml['y_rmse']:.4f}}}",
        f"\\newcommand{{\\TreatmentRtwo}}{{{dml['d_r2']:.3f}}}",
    ]
    (RESULTS / "analysis_numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    panel = build_panel()
    controls = [
        "population",
        "pop_growth",
        "lag_zori",
    ]
    results = {
        "fe": ols_fe(panel, []),
        "fe_controls": ols_fe(panel, controls),
        "dml_rf": dml_plr(panel, trim=False, learner_name="rf"),
        "dml_trim": dml_plr(panel, trim=True, learner_name="rf"),
        "dml_gb": dml_plr(panel, trim=False, learner_name="gb"),
        "dml_lasso": dml_plr(panel, trim=False, learner_name="lasso"),
        "heterogeneity": ols_fe(panel, controls, interaction=True),
    }
    serializable = json.loads(json.dumps(results, default=lambda o: o.tolist() if hasattr(o, "tolist") else o))
    (RESULTS / "analysis_results.json").write_text(json.dumps(serializable, indent=2))
    write_tex(results, panel)
    make_figures(panel, results)
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
