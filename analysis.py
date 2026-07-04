"""
Core econometric engine: resolve a company name to an NSE ticker, pull five
years of daily prices for it and the Nifty 50, run the same OLS market-model
workflow as the original R case study, and render the charts + narrative text
used by both the HTML report and the downloadable PDF.
"""
import base64
import io
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan

# ---- Design system (matches the original R report) -------------------------
COL_BLUE = "#2a78d6"     # Nifty 50 (categorical slot 1)
COL_ORANGE = "#eb6834"   # Company  (categorical slot 8)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10.5,
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


class AnalysisError(Exception):
    pass


# ------------------------------------------------------------------ ticker --
def resolve_ticker(query: str):
    """Best-effort resolution of a free-text company name to an NSE ticker.
    Returns (ticker, display_name). Raises AnalysisError if nothing works."""
    q = query.strip()
    if not q:
        raise AnalysisError("Please enter a company name or ticker symbol.")

    # 1. If it looks like a bare ticker (no spaces), try it directly first -
    #    covers common short codes (ITC, UPL, TCS, SBIN...) that Yahoo's
    #    fuzzy search sometimes fails to surface among global collisions.
    compact = re.sub(r"[^A-Za-z0-9&\-]", "", q).upper()
    if compact and " " not in q:
        guess = f"{compact}.NS"
        try:
            hist = yf.Ticker(guess).history(period="5d")
        except Exception:
            hist = None
        if hist is not None and not hist.empty:
            name = q
            try:
                info = yf.Ticker(guess).get_info()
                name = info.get("longName") or info.get("shortName") or q
            except Exception:
                pass
            return guess, name

    # 2. Fuzzy company-name search, restricted to NSE ("NSI") equities.
    try:
        quotes = yf.Search(q, max_results=10).quotes
    except Exception:
        quotes = []
    candidates = [x for x in quotes if x.get("exchange") == "NSI" and x.get("quoteType") == "EQUITY"]
    if candidates:
        top = candidates[0]
        return top["symbol"], top.get("longname") or top.get("shortname") or q

    # 3. Last resort: compacted name + .NS (handles single-word names the
    #    ticker-guess step above skipped because they contained no spaces
    #    check, e.g. shouldn't normally reach here, but cheap to try).
    if compact:
        guess = f"{compact}.NS"
        try:
            hist = yf.Ticker(guess).history(period="5d")
        except Exception:
            hist = None
        if hist is not None and not hist.empty:
            return guess, q

    raise AnalysisError(
        f'Could not find an NSE-listed company matching "{query}". '
        "Try the full company name (e.g. 'Tata Motors') or its NSE ticker (e.g. 'TATAMOTORS')."
    )


# -------------------------------------------------------------------- data --
def _get_close_series(ticker: str, period="5y") -> pd.Series:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if df.empty:
        raise AnalysisError(f"No price data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"].iloc[:, 0]
    else:
        s = df["Close"]
    s.name = ticker
    return s.dropna()


def build_dataset(ticker: str) -> pd.DataFrame:
    nifty = _get_close_series("^NSEI").rename("Nifty_Close")
    stock = _get_close_series(ticker).rename("Stock_Close")
    df = pd.concat([nifty, stock], axis=1, join="inner").dropna()
    if len(df) < 60:
        raise AnalysisError(
            "Fewer than 60 overlapping trading days of data are available for this "
            "company against the Nifty 50 - it may be too newly listed for this analysis."
        )
    df["Nifty_ret"] = np.log(df["Nifty_Close"]).diff() * 100
    df["Stock_ret"] = np.log(df["Stock_Close"]).diff() * 100
    df = df.iloc[1:]
    df["Nifty_ret_lag1"] = df["Nifty_ret"].shift(1)
    df = df.iloc[1:]
    return df


# --------------------------------------------------------------- formatting --
def fmt(x, d=3):
    return f"{x:,.{d}f}"


def fmt_p(p):
    return "< 0.001" if p < 0.001 else fmt(p, 3)


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def png_to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


# ------------------------------------------------------------------ charts --
def chart_prices(df, company_name):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6), sharex=False)
    axes[0].plot(df.index, df["Nifty_Close"], color=COL_BLUE, linewidth=1.3)
    axes[0].set_title("Nifty 50", loc="left", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    axes[0].set_ylabel("Index level")
    axes[1].plot(df.index, df["Stock_Close"], color=COL_ORANGE, linewidth=1.3)
    axes[1].set_title(company_name, loc="left", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    axes[1].set_ylabel("Price (Rs.)")
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def chart_indexed(df, company_name):
    idx_nifty = df["Nifty_Close"] / df["Nifty_Close"].iloc[0] * 100
    idx_stock = df["Stock_Close"] / df["Stock_Close"].iloc[0] * 100
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.axhline(100, color=BASELINE, linewidth=0.9, linestyle="--")
    ax.plot(df.index, idx_nifty, color=COL_BLUE, linewidth=1.4, label="Nifty 50")
    ax.plot(df.index, idx_stock, color=COL_ORANGE, linewidth=1.4, label=company_name)
    ax.set_ylabel("Cumulative growth (start = 100)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def chart_histograms(df, company_name):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].hist(df["Nifty_ret"], bins=50, color=COL_BLUE, alpha=0.85)
    axes[0].set_title("Nifty 50 daily return (%)", loc="left", fontsize=10.5, fontweight="bold")
    axes[1].hist(df["Stock_ret"], bins=50, color=COL_ORANGE, alpha=0.85)
    axes[1].set_title(f"{company_name} daily return (%)", loc="left", fontsize=10.5, fontweight="bold")
    for ax in axes:
        ax.set_ylabel("Count")
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def chart_scatter(df, model_simple, company_name):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = df["Nifty_ret"].values
    y = df["Stock_ret"].values
    ax.scatter(x, y, alpha=0.35, s=14, color=COL_BLUE, linewidths=0)
    xs = np.linspace(x.min(), x.max(), 100)
    Xp = sm.add_constant(xs)
    pred = model_simple.get_prediction(Xp).summary_frame(alpha=0.05)
    ax.plot(xs, pred["mean"], color=COL_ORANGE, linewidth=1.8)
    ax.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"], color=COL_ORANGE, alpha=0.15)
    ax.set_xlabel("Nifty 50 daily return (%)")
    ax.set_ylabel(f"{company_name} daily return (%)")
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def chart_diagnostics(model_multi):
    resid = model_multi.resid
    fitted = model_multi.fittedvalues
    std_resid = model_multi.get_influence().resid_studentized_internal

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))

    axes[0, 0].axhline(0, color=BASELINE, linewidth=0.9)
    axes[0, 0].scatter(fitted, resid, alpha=0.35, s=12, color=COL_BLUE, linewidths=0)
    axes[0, 0].set_title("Residuals vs Fitted", loc="left", fontsize=10.5, fontweight="bold")
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")

    sm.qqplot(std_resid, line="45", ax=axes[0, 1], markerfacecolor=COL_BLUE, markeredgecolor=COL_BLUE, alpha=0.4)
    axes[0, 1].get_lines()[1].set_color(COL_ORANGE)
    axes[0, 1].set_title("Normal Q-Q", loc="left", fontsize=10.5, fontweight="bold")

    axes[1, 0].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.35, s=12, color=COL_BLUE, linewidths=0)
    axes[1, 0].set_title("Scale-Location", loc="left", fontsize=10.5, fontweight="bold")
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel("sqrt(|Std. residuals|)")

    axes[1, 1].hist(resid, bins=45, color=COL_BLUE, alpha=0.85)
    axes[1, 1].set_title("Residual Distribution", loc="left", fontsize=10.5, fontweight="bold")
    axes[1, 1].set_xlabel("Residuals")
    axes[1, 1].set_ylabel("Count")

    fig.tight_layout()
    return fig_to_png_bytes(fig)


def chart_prediction(scenario_df):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = scenario_df["Nifty_ret"]
    ax.fill_between(x, scenario_df["mean_ci_lower"], scenario_df["mean_ci_upper"], color=COL_ORANGE, alpha=0.15)
    ax.plot(x, scenario_df["mean"], color=COL_ORANGE, linewidth=1.8)
    ax.scatter(x, scenario_df["mean"], color=COL_BLUE, s=40, zorder=5)
    ax.set_xlabel("Hypothetical Nifty 50 daily return (%)")
    ax.set_ylabel("Predicted company daily return (%)")
    fig.tight_layout()
    return fig_to_png_bytes(fig)


# ------------------------------------------------------------ narrative text --
def _beta_desc(beta):
    if beta > 1.1:
        return f"amplified Nifty 50 moves, swinging about {(beta - 1) * 100:.0f}% more than the market on an average day"
    if beta < 0.9:
        return f"dampened Nifty 50 moves, swinging about {(1 - beta) * 100:.0f}% less than the market on an average day"
    return "moved roughly in line with the market, one-for-one with Nifty 50 swings on an average day"


def build_report(query: str) -> dict:
    ticker, company_name = resolve_ticker(query)
    df = build_dataset(ticker)
    n_obs = len(df)
    date_range = (df.index.min(), df.index.max())

    # ---- Models ----
    X1 = sm.add_constant(df["Nifty_ret"])
    model_simple = sm.OLS(df["Stock_ret"], X1).fit()

    X2 = sm.add_constant(df[["Nifty_ret", "Nifty_ret_lag1"]])
    model_multi = sm.OLS(df["Stock_ret"], X2).fit()

    ci_simple = model_simple.conf_int(alpha=0.05)
    ci_multi = model_multi.conf_int(alpha=0.05)

    beta_hat = model_simple.params["Nifty_ret"]
    se_beta = model_simple.bse["Nifty_ret"]
    t_beta1 = (beta_hat - 1) / se_beta
    p_beta1 = 2 * stats.t.sf(abs(t_beta1), model_simple.df_resid)

    anova_res = anova_lm(model_simple, model_multi)
    f_stat = anova_res["F"].iloc[1]
    f_pval = anova_res["Pr(>F)"].iloc[1]

    bp_stat, bp_pval, _, _ = het_breuschpagan(model_multi.resid, model_multi.model.exog)

    robust = model_multi.get_robustcov_results(cov_type="HC1")

    scenarios = pd.DataFrame({"Nifty_ret": [-2, -1, 0, 1, 2], "Nifty_ret_lag1": [0, 0, 0, 0, 0]})
    Xs = sm.add_constant(scenarios, has_constant="add")
    pred_frame = model_multi.get_prediction(Xs).summary_frame(alpha=0.05)
    pred_frame["Nifty_ret"] = scenarios["Nifty_ret"].values

    corr = df[["Nifty_ret", "Stock_ret"]].corr().iloc[0, 1]

    # ---- Charts ----
    images = {
        "prices": chart_prices(df, company_name),
        "indexed": chart_indexed(df, company_name),
        "hist": chart_histograms(df, company_name),
        "scatter": chart_scatter(df, model_simple, company_name),
        "diagnostics": chart_diagnostics(model_multi),
        "prediction": chart_prediction(pred_frame),
    }

    # ---- Tables ----
    def reg_table(model, ci):
        rows = []
        for term in model.params.index:
            rows.append({
                "Term": term,
                "Estimate": fmt(model.params[term]),
                "Std. Error": fmt(model.bse[term]),
                "t value": fmt(model.tvalues[term], 2),
                "p value": fmt_p(model.pvalues[term]),
                "95% CI": f"[{fmt(ci.loc[term, 0])}, {fmt(ci.loc[term, 1])}]",
            })
        return rows

    reg_simple_tbl = reg_table(model_simple, ci_simple)
    reg_multi_tbl = reg_table(model_multi, ci_multi)

    robust_tbl = []
    for i, term in enumerate(model_multi.params.index):
        robust_tbl.append({
            "Term": term,
            "Estimate": fmt(robust.params[i]),
            "Robust SE": fmt(robust.bse[i]),
            "t value": fmt(robust.tvalues[i], 2),
            "p value": fmt_p(robust.pvalues[i]),
        })

    compare_tbl = []
    for i, term in enumerate(model_multi.params.index):
        compare_tbl.append({
            "Term": term,
            "OLS Std. Error": fmt(model_multi.bse[term]),
            "Robust SE (HC1)": fmt(robust.bse[i]),
        })

    stat_tbl = [
        {"Series": "Nifty 50 return (%)", "Mean": fmt(df["Nifty_ret"].mean()), "SD": fmt(df["Nifty_ret"].std()),
         "Min": fmt(df["Nifty_ret"].min()), "Max": fmt(df["Nifty_ret"].max()), "Corr. with other": fmt(corr)},
        {"Series": f"{company_name} return (%)", "Mean": fmt(df["Stock_ret"].mean()), "SD": fmt(df["Stock_ret"].std()),
         "Min": fmt(df["Stock_ret"].min()), "Max": fmt(df["Stock_ret"].max()), "Corr. with other": fmt(corr)},
    ]

    pred_tbl = []
    for _, row in pred_frame.iterrows():
        pred_tbl.append({
            "Nifty 50 move (%)": fmt(row["Nifty_ret"], 1),
            "Predicted move (%)": fmt(row["mean"]),
            "95% CI lower": fmt(row["mean_ci_lower"]),
            "95% CI upper": fmt(row["mean_ci_upper"]),
        })

    meta_tbl = [
        {"Field": "Dependent series", "Value": f"{company_name} ({ticker}) daily log return, %"},
        {"Field": "Independent series", "Value": "Nifty 50 (^NSEI) daily log return, %"},
        {"Field": "Frequency", "Value": "Daily (trading days)"},
        {"Field": "Sample period", "Value": f"{date_range[0]:%d %b %Y} - {date_range[1]:%d %b %Y}"},
        {"Field": "Observations", "Value": str(n_obs)},
        {"Field": "Source", "Value": "Yahoo Finance"},
    ]

    # ---- Narrative text (explains every table/chart, in plain English) ----
    text = {}

    text["intro"] = (
        f"This report estimates the market beta of {company_name} ({ticker}) against the Nifty 50 "
        f"benchmark index using {n_obs:,} trading days of daily price data spanning "
        f"{date_range[0]:%d %b %Y} to {date_range[1]:%d %b %Y}. The analysis follows the standard "
        "econometric workflow for a market-model regression: exploratory analysis of returns, a simple "
        "OLS regression of the company's return on the market's return, an extended specification that "
        "adds yesterday's market return, hypothesis tests on the estimated beta, residual diagnostics, "
        "and heteroskedasticity-robust inference."
    )

    text["prices"] = (
        f"The panels above show five years of daily closing levels for the Nifty 50 and {company_name} on "
        "independent y-axes. An index and a stock are not comparable in level terms - one is measured in "
        "index points, the other in rupees - so each series is plotted on its own scale rather than a "
        "shared or dual axis, which would invite a misleading visual comparison."
    )

    idx_nifty_end = df["Nifty_Close"].iloc[-1] / df["Nifty_Close"].iloc[0] * 100
    idx_stock_end = df["Stock_Close"].iloc[-1] / df["Stock_Close"].iloc[0] * 100
    outperform = "outperformed" if idx_stock_end > idx_nifty_end else "underperformed"
    text["indexed"] = (
        f"Both series are rebased to 100 on their first common trading day, which puts them on one honest, "
        f"shared axis. Over the sample, the Nifty 50 grew to an index value of {idx_nifty_end:,.0f} while "
        f"{company_name} reached {idx_stock_end:,.0f} - meaning {company_name} {outperform} the benchmark "
        "in total cumulative growth terms over this window. Note that cumulative outperformance is a "
        "separate question from beta: a stock can beat the index over a period while still moving less "
        "(or more) than the index on a typical single day, which is what the regression below measures."
    )

    text["stats"] = (
        f"The table summarizes daily log returns over the full sample. {company_name}'s standard deviation "
        f"({fmt(df['Stock_ret'].std())}%) is {'higher' if df['Stock_ret'].std() > df['Nifty_ret'].std() else 'lower'} "
        f"than the Nifty 50's ({fmt(df['Nifty_ret'].std())}%), which is typical: a single stock carries "
        "idiosyncratic, company-specific risk on top of whatever market-wide risk it shares with the index, "
        f"while the index's diversification smooths out single-stock noise. The correlation between the two "
        f"return series is {fmt(corr)}, indicating a "
        f"{'strong' if abs(corr) > 0.6 else 'moderate' if abs(corr) > 0.3 else 'weak'} linear relationship "
        "between daily market moves and daily company moves - this correlation is what the regression below "
        "converts into a precise, interpretable slope. The histograms show both return distributions are "
        "roughly bell-shaped and centered near zero, with fatter tails than a normal distribution - the "
        "occasional large daily swing that is characteristic of financial return data."
    )

    text["scatter"] = (
        f"Each point is one trading day. The fitted line is the OLS market model: "
        f"{company_name} return = {fmt(model_simple.params['const'])} + {fmt(beta_hat)} x Nifty return, with "
        f"an R-squared of {fmt(model_simple.rsquared)}. The slope, {fmt(beta_hat)}, is the estimated market "
        f"beta: on an average day, a 1% move in the Nifty 50 is associated with a {fmt(beta_hat)}% move in "
        f"{company_name}. The shaded band is the 95% confidence interval for the fitted mean response, not "
        "for individual days - it is narrow near the center of the data and widens toward the extremes "
        "because the regression line is estimated more precisely near the average market return."
    )

    text["regression"] = (
        f"Model 1 is the simple market model, {company_name} return on the same-day Nifty return alone. "
        "Model 2 adds yesterday's Nifty return, testing whether the stock partly reacts to the market with "
        "a one-day lag - a symptom of non-synchronous or thin trading. In both tables, the p-value tests "
        "whether each coefficient is statistically distinguishable from zero, and the 95% CI gives a range "
        "of plausible values for the true coefficient given this sample."
    )

    reject1 = p_beta1 < 0.05
    text["hyp1"] = (
        f"Testing H0: beta = 1 (does the stock move one-for-one with the market on average?) gives "
        f"t = {fmt(t_beta1, 3)}, p = {fmt(p_beta1, 3)}. At the 5% significance level we "
        f"{'reject' if reject1 else 'fail to reject'} H0: {company_name}'s beta "
        f"{'is' if reject1 else 'is not'} significantly different from 1. "
        f"The point estimate of {fmt(beta_hat)} suggests the stock has historically {_beta_desc(beta_hat)}."
    )

    reject_f = f_pval < 0.05
    text["hyp2"] = (
        f"An F-test comparing Model 1 against Model 2 asks whether yesterday's Nifty return adds "
        f"explanatory power beyond today's: F = {fmt(f_stat, 3)}, p = {fmt(f_pval, 3)}. We "
        f"{'reject' if reject_f else 'fail to reject'} the null that the lagged term has no effect, so the "
        f"lagged market return {'does add' if reject_f else 'adds little'} statistically significant "
        f"explanatory power in this sample"
        + (", pointing to mild non-synchronous trading effects." if reject_f else ".")
    )

    het_present = bp_pval < 0.05
    text["diagnostics"] = (
        f"The four panels check the assumptions behind Model 2's OLS estimates. Residuals vs Fitted and "
        "Scale-Location should show no systematic pattern or funnel shape if error variance is constant; "
        "the Normal Q-Q plot should hug the 45-degree line if residuals are approximately normal; the "
        "histogram shows the shape of the residual distribution directly. "
        f"A Breusch-Pagan test for heteroskedasticity gives BP = {fmt(bp_stat, 3)}, p = {fmt_p(bp_pval)}. "
        + (
            "The null of constant error variance is rejected: heteroskedasticity is present, which is "
            "routine in daily return data due to volatility clustering (calm periods followed by turbulent "
            "ones). This motivates the heteroskedasticity-robust standard errors that follow."
            if het_present else
            "The null of constant error variance is not rejected in this sample, though robust standard "
            "errors are still reported next as routine practice for return regressions, which often show "
            "volatility clustering in other sub-periods."
        )
    )

    text["robust"] = (
        "Point estimates are identical to Model 2 above - robust standard errors only recompute the "
        "uncertainty around those estimates so that t-values and p-values remain valid even if error "
        "variance is not constant across observations. Compare the OLS and robust standard errors in the "
        "second table: "
        + (
            "they differ enough here to matter for inference, which is consistent with the heteroskedasticity "
            "detected above."
            if abs(robust.bse[1] - model_multi.bse.iloc[1]) / model_multi.bse.iloc[1] > 0.05 else
            "they are close in this sample, so heteroskedasticity has only a mild effect on inference here, "
            "though robust errors remain the safer default for return data."
        )
    )

    text["prediction"] = (
        "Using Model 2, this table projects the company's expected daily return for a range of hypothetical "
        "same-day Nifty 50 moves, holding yesterday's Nifty return at its typical value of zero. The shaded "
        "band is the 95% confidence interval for the predicted mean return at each hypothetical market move - "
        "it reflects uncertainty in the fitted relationship, not the much wider range of actual single-day "
        "outcomes an investor might observe."
    )

    text["conclusion"] = (
        f"{company_name}'s estimated market beta is {fmt(beta_hat, 2)}, meaning the stock has historically "
        f"{_beta_desc(beta_hat)}. The hypothesis that beta equals 1 is "
        f"{'rejected' if reject1 else 'not rejected'} at the 5% level. Adding yesterday's Nifty return to "
        f"the model {'adds' if reject_f else 'does not add'} statistically significant explanatory power, "
        f"suggesting {'some' if reject_f else 'little'} evidence of non-synchronous trading effects in this "
        f"sample. {'Heteroskedasticity was detected, so the robust standard errors above should be used for inference.' if het_present else 'No significant heteroskedasticity was detected, though robust standard errors were reported as routine practice.'} "
        f"Overall, the market model explains {fmt(model_simple.rsquared * 100, 1)}% of the day-to-day "
        f"variation in {company_name}'s returns (R-squared), leaving the majority of its daily moves "
        "attributable to company-specific rather than market-wide factors."
    )

    return {
        "ticker": ticker,
        "company_name": company_name,
        "n_obs": n_obs,
        "date_range": date_range,
        "generated": datetime.now(),
        "images": images,
        "tables": {
            "meta": meta_tbl,
            "stats": stat_tbl,
            "reg_simple": reg_simple_tbl,
            "reg_multi": reg_multi_tbl,
            "robust": robust_tbl,
            "compare": compare_tbl,
            "prediction": pred_tbl,
        },
        "text": text,
        "stats_raw": {
            "beta_hat": beta_hat,
            "r2_simple": model_simple.rsquared,
            "t_beta1": t_beta1,
            "p_beta1": p_beta1,
            "f_stat": f_stat,
            "f_pval": f_pval,
            "bp_stat": bp_stat,
            "bp_pval": bp_pval,
        },
    }
