"""
Core econometric engine: resolve a company name to an NSE ticker, pull five
years of daily prices for it and the Nifty 50, run the same OLS market-model
workflow as the original R case study, and render the charts + narrative text
used by both the HTML report and the downloadable PDF.
"""
import base64
import io
import re
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from arch import arch_model
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning, module="arch")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

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


def chart_acf_pacf(df, company_name, nlags=20):
    ret = df["Stock_ret"].values
    acf_vals = acf(ret, nlags=nlags, fft=True)
    pacf_vals = pacf(ret, nlags=nlags, method="ywm")
    band = 1.96 / np.sqrt(len(ret))
    lags = np.arange(nlags + 1)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, vals, color, title in (
        (axes[0], acf_vals, COL_BLUE, f"{company_name} return: ACF"),
        (axes[1], pacf_vals, COL_ORANGE, f"{company_name} return: PACF"),
    ):
        ax.bar(lags[1:], vals[1:], width=0.35, color=color)
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        ax.axhline(band, color=BASELINE, linewidth=0.9, linestyle="--")
        ax.axhline(-band, color=BASELINE, linewidth=0.9, linestyle="--")
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Lag (trading days)")
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def chart_arima_forecast(df, company_name, forecast_dates, fc_mean, fc_lower, fc_upper):
    history = df["Stock_ret"].iloc[-60:]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.axhline(0, color=BASELINE, linewidth=0.8)
    ax.plot(history.index, history.values, color=COL_BLUE, linewidth=1.1, label="Observed return")
    ax.plot(forecast_dates, fc_mean, color=COL_ORANGE, linewidth=1.8, label="Forecast")
    ax.fill_between(forecast_dates, fc_lower, fc_upper, color=COL_ORANGE, alpha=0.15)
    ax.axvline(history.index[-1], color=INK_MUTED, linewidth=0.8, linestyle=":")
    ax.set_ylabel(f"{company_name} daily return (%)")
    ax.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def chart_garch_volatility(df, cond_vol, forecast_dates, forecast_vol, company_name):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(df.index, cond_vol, color=COL_BLUE, linewidth=1.0, label="Estimated daily volatility (in-sample)")
    ax.plot(forecast_dates, forecast_vol, color=COL_ORANGE, linewidth=1.8, linestyle="--", label="Forecast volatility")
    ax.axvline(df.index[-1], color=INK_MUTED, linewidth=0.8, linestyle=":")
    ax.set_ylabel(f"{company_name} conditional daily volatility (%)")
    ax.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig_to_png_bytes(fig)


# --------------------------------------------------- stationarity & forecasting --
def adf_test(series, regression="c"):
    stat, pval, usedlag, nobs, _crit, _icbest = adfuller(series.dropna().values, autolag="AIC", regression=regression)
    return {"stat": stat, "pval": pval, "usedlag": usedlag, "nobs": nobs}


def select_arima(returns, max_p=3, max_q=3):
    """Grid-search ARMA(p,0,q) orders (0..max_p, 0..max_q) and keep the one
    that minimizes BIC. BIC (rather than AIC) is used deliberately: on noisy
    daily-return data, AIC tends to pick needlessly complex ARMA models whose
    AR and MA roots nearly cancel, while BIC favors the simplest model
    consistent with the data - including the "no structure" ARIMA(0,0,0)
    constant-mean model when returns are indistinguishable from white noise."""
    best = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                fitted = ARIMA(returns, order=(p, 0, q), trend="c").fit()
            except Exception:
                continue
            if best is None or fitted.bic < best[0].bic:
                best = (fitted, p, q)
    return best


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

    # ---- Stationarity & autocorrelation ----
    adf_nifty_level = adf_test(df["Nifty_Close"], regression="ct")
    adf_stock_level = adf_test(df["Stock_Close"], regression="ct")
    adf_nifty_ret = adf_test(df["Nifty_ret"], regression="c")
    adf_stock_ret = adf_test(df["Stock_ret"], regression="c")

    lb_raw = acorr_ljungbox(df["Stock_ret"], lags=[10], return_df=True)
    lb_stat = lb_raw["lb_stat"].iloc[0]
    lb_pval = lb_raw["lb_pvalue"].iloc[0]

    # ---- ARIMA return forecast ----
    horizon = 10
    arima_model, arima_p, arima_q = select_arima(df["Stock_ret"])
    arima_fc = arima_model.get_forecast(steps=horizon)
    arima_mean = np.asarray(arima_fc.predicted_mean)
    arima_ci = np.asarray(arima_fc.conf_int(alpha=0.05))
    forecast_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon)

    last_price = df["Stock_Close"].iloc[-1]
    implied_price = last_price * np.exp(np.cumsum(arima_mean) / 100)

    lb_arima = acorr_ljungbox(arima_model.resid, lags=[10], return_df=True)
    lb_arima_stat = lb_arima["lb_stat"].iloc[0]
    lb_arima_pval = lb_arima["lb_pvalue"].iloc[0]

    # ---- GARCH(1,1) volatility model ----
    garch_res = arch_model(df["Stock_ret"], mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    garch_alpha = garch_res.params["alpha[1]"]
    garch_beta = garch_res.params["beta[1]"]
    garch_persistence = garch_alpha + garch_beta
    garch_half_life = (
        np.log(0.5) / np.log(garch_persistence) if 0 < garch_persistence < 1 else float("nan")
    )
    garch_fc = garch_res.forecast(horizon=horizon, reindex=False)
    garch_fc_vol = np.sqrt(garch_fc.variance.values[-1])
    garch_unconditional_vol = np.sqrt(
        garch_res.params["omega"] / (1 - garch_persistence)
    ) if garch_persistence < 1 else float("nan")

    # ---- Charts ----
    images = {
        "prices": chart_prices(df, company_name),
        "indexed": chart_indexed(df, company_name),
        "hist": chart_histograms(df, company_name),
        "scatter": chart_scatter(df, model_simple, company_name),
        "diagnostics": chart_diagnostics(model_multi),
        "prediction": chart_prediction(pred_frame),
        "acf_pacf": chart_acf_pacf(df, company_name),
        "arima_forecast": chart_arima_forecast(
            df, company_name, forecast_dates, arima_mean, arima_ci[:, 0], arima_ci[:, 1]
        ),
        "garch_vol": chart_garch_volatility(
            df, garch_res.conditional_volatility, forecast_dates, garch_fc_vol, company_name
        ),
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

    def adf_row(label, res):
        stationary = res["pval"] < 0.05
        return {
            "Series": label,
            "ADF statistic": fmt(res["stat"], 3),
            "p value": fmt_p(res["pval"]),
            "Lags used": str(res["usedlag"]),
            "Conclusion": "Stationary (rejects unit root)" if stationary else "Non-stationary (unit root present)",
        }

    stationarity_tbl = [
        adf_row("Nifty 50 price level", adf_nifty_level),
        adf_row(f"{company_name} price level", adf_stock_level),
        adf_row("Nifty 50 daily return", adf_nifty_ret),
        adf_row(f"{company_name} daily return", adf_stock_ret),
    ]

    autocorr_tbl = [{
        "Test": "Ljung-Box, 10 lags (raw returns)",
        "Statistic": fmt(lb_stat, 3),
        "p value": fmt_p(lb_pval),
        "Conclusion": "Significant autocorrelation detected" if lb_pval < 0.05 else "No significant autocorrelation detected",
    }]

    arima_coef_tbl = []
    for term in arima_model.params.index:
        if term == "sigma2":
            continue
        arima_coef_tbl.append({
            "Term": term,
            "Estimate": fmt(arima_model.params[term]),
            "Std. Error": fmt(arima_model.bse[term]),
            "t value": fmt(arima_model.tvalues[term], 2),
            "p value": fmt_p(arima_model.pvalues[term]),
        })

    arima_forecast_tbl = []
    for i in range(horizon):
        arima_forecast_tbl.append({
            "Date": f"{forecast_dates[i]:%d %b %Y}",
            "Forecast return (%)": fmt(arima_mean[i]),
            "95% CI lower": fmt(arima_ci[i, 0]),
            "95% CI upper": fmt(arima_ci[i, 1]),
            "Implied price (Rs.)": fmt(implied_price[i], 2),
        })

    arima_diag_tbl = [{
        "Test": "Ljung-Box, 10 lags (ARIMA residuals)",
        "Statistic": fmt(lb_arima_stat, 3),
        "p value": fmt_p(lb_arima_pval),
        "Conclusion": "Residual autocorrelation remains" if lb_arima_pval < 0.05 else "No residual autocorrelation - model is adequate",
    }]

    garch_coef_tbl = []
    for term in garch_res.params.index:
        garch_coef_tbl.append({
            "Term": term,
            "Estimate": fmt(garch_res.params[term], 4),
            "Std. Error": fmt(garch_res.std_err[term], 4),
            "p value": fmt_p(garch_res.pvalues[term]),
        })
    garch_coef_tbl.append({
        "Term": "alpha + beta (persistence)",
        "Estimate": fmt(garch_persistence, 4),
        "Std. Error": "-",
        "p value": "-",
    })

    garch_forecast_tbl = []
    for i in range(horizon):
        garch_forecast_tbl.append({
            "Date": f"{forecast_dates[i]:%d %b %Y}",
            "Forecast daily volatility (%)": fmt(garch_fc_vol[i]),
            "Annualized volatility (%)": fmt(garch_fc_vol[i] * np.sqrt(252), 1),
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
        f"{date_range[0]:%d %b %Y} to {date_range[1]:%d %b %Y}. The analysis combines a cross-sectional "
        "market-model regression with a dedicated time-series toolkit: exploratory analysis of returns, a "
        "simple OLS regression of the company's return on the market's return, an extended specification "
        "that adds yesterday's market return, hypothesis tests on the estimated beta, residual diagnostics, "
        "and heteroskedasticity-robust inference, followed by unit-root and autocorrelation tests, an "
        "ARIMA model of the return series with a forecast for the next 10 trading days, and a GARCH(1,1) "
        "model of the stock's time-varying volatility."
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

    text["stationarity"] = (
        "ARIMA and GARCH models assume the series being modeled is stationary - its mean and variance don't "
        "systematically drift over time. The table above reports Augmented Dickey-Fuller (ADF) tests, which "
        "test the null hypothesis that a series contains a unit root (is non-stationary). As is typical for "
        f"equity prices, both the Nifty 50 and {company_name}'s price level fail to reject the unit-root null "
        f"(p = {fmt_p(adf_nifty_level['pval'])} and p = {fmt_p(adf_stock_level['pval'])} respectively): prices "
        "wander over time with no fixed level to revert to, which is exactly why analysts model returns "
        f"rather than price levels. Daily log returns for both series comfortably reject the unit-root null "
        f"(p = {fmt_p(adf_nifty_ret['pval'])} and p = {fmt_p(adf_stock_ret['pval'])}), confirming returns are "
        "stationary and therefore valid inputs for the ARIMA and GARCH models that follow."
    )

    lb_significant = lb_pval < 0.05
    text["autocorr"] = (
        f"The autocorrelation function (ACF) and partial autocorrelation function (PACF) above show how "
        f"strongly {company_name}'s return on a given day is correlated with its own return 1 to 20 trading "
        "days earlier. Bars crossing the dashed bands are statistically distinguishable from zero at the 5% "
        f"level. A Ljung-Box test formally checks whether the first 10 lags are jointly significant: "
        f"Q = {fmt(lb_stat, 3)}, p = {fmt_p(lb_pval)}. "
        + (
            "Significant autocorrelation is present, meaning some of the stock's own recent return history "
            "helps explain its next move - the ARIMA model below is fit to capture that structure."
            if lb_significant else
            "No significant autocorrelation is detected, meaning the stock's own recent return history carries "
            "little information about its next move on its own - a first hint that little short-horizon "
            "structure remains for the ARIMA model below to find."
        )
    )

    arima_is_whitenoise = (arima_p == 0 and arima_q == 0)
    arima_order_str = f"ARIMA({arima_p}, 0, {arima_q})"
    text["arima"] = (
        "An ARIMA(p, d, q) model forecasts a series from its own past values (the AR terms) and past forecast "
        "errors (the MA terms); d = 0 here because the return series is already stationary, as shown above. "
        "Every combination of p and q from 0 to 3 was fit and compared using the Bayesian Information "
        "Criterion (BIC), which penalizes unnecessary complexity more heavily than the more common AIC and "
        "so avoids fitting noise - a real risk with daily returns, where AIC often selects AR/MA terms whose "
        f"roots nearly cancel out. The winning specification is {arima_order_str}. "
        + (
            f"This is the model with no AR or MA terms at all - just a constant mean. In plain terms, this is "
            f"direct evidence that {company_name}'s daily returns are statistically indistinguishable from "
            "white noise: past returns contain no statistically useful information for forecasting tomorrow's "
            "return, consistent with the weak form of market efficiency."
            if arima_is_whitenoise else
            f"Its coefficients are reported in the table above. A Ljung-Box test on the model's residuals "
            f"(Q = {fmt(lb_arima_stat, 3)}, p = {fmt_p(lb_arima_pval)}) checks whether any autocorrelation "
            "remains unexplained - "
            + (
                "some does, meaning even the selected model does not fully capture the return dynamics."
                if lb_arima_pval < 0.05 else
                "none remains, indicating the model adequately captures the autocorrelation structure in the data."
            )
        )
    )

    text["arima_forecast"] = (
        f"The table and chart project {company_name}'s daily return for the next {horizon} trading days, "
        f"together with an implied price path compounded forward from the last observed close of "
        f"Rs. {fmt(last_price, 2)}. "
        + (
            f"Because the selected model contains no predictive structure, the point forecast sits at the "
            f"sample's constant mean return ({fmt(arima_model.params['const'])}% per day) for every horizon, "
            "and the 95% confidence band stays wide and roughly constant from day one - an honest reflection "
            "of the fact that short-horizon returns are not meaningfully predictable from their own history "
            "alone. The implied price path is therefore a near-random walk around the last observed price, "
            "not a directional prediction."
            if arima_is_whitenoise else
            "The point forecast reflects the fitted AR/MA dynamics and converges toward the model's long-run "
            "mean return as the horizon extends, while the 95% confidence band widens with each additional "
            "day, reflecting compounding forecast uncertainty. As with any short-horizon return forecast, "
            "the band is wide relative to the point estimate - a reminder that day-to-day direction remains "
            "hard to call even when weak statistical structure is present."
        )
    )

    garch_persistent = 0 < garch_persistence < 1
    text["garch"] = (
        "The Breusch-Pagan test earlier flagged heteroskedasticity in the regression residuals - error "
        "variance that changes over time. A GARCH(1,1) model characterizes that time-varying volatility "
        "directly, rather than only correcting standard errors around it. It models today's variance as a "
        "weighted combination of a long-run average, yesterday's squared surprise "
        f"(the ARCH term, alpha = {fmt(garch_alpha, 3)}), and yesterday's variance (the GARCH term, "
        f"beta = {fmt(garch_beta, 3)}). Their sum, alpha + beta = {fmt(garch_persistence, 3)}, measures "
        "volatility persistence - how long a shock to volatility takes to fade. "
        + (
            f"With persistence below 1, a shock here has an estimated half-life of about "
            f"{garch_half_life:.1f} trading days before decaying halfway back to its long-run average of "
            f"roughly {fmt(garch_unconditional_vol, 2)}% daily volatility."
            if garch_persistent else
            "Persistence is at or above 1, indicating volatility shocks decay very slowly, or not at all, in "
            "this sample - a near-integrated GARCH process where turbulent periods can persist for a long time."
        )
        + " The chart shows the model's estimated daily volatility across the full sample - the visible "
        "clustering of calm and turbulent stretches is exactly the pattern the earlier Breusch-Pagan test was "
        f"picking up - together with a {horizon}-day forecast that "
        + (
            "reverts toward the long-run average as the horizon extends, the hallmark of a mean-reverting "
            "volatility process."
            if garch_persistent else
            "stays close to its current level across the forecast window, reflecting the slow-decaying "
            "persistence estimated above."
        )
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
        "attributable to company-specific rather than market-wide factors. "
        + (
            f"On the time-series side, returns are stationary while price levels are not, {company_name}'s "
            f"own return history shows {'some' if lb_significant else 'little'} exploitable autocorrelation, "
            f"and the BIC-selected {arima_order_str} model "
            + (
                "found no predictable structure at all - reinforcing that short-horizon return direction is "
                "very difficult to forecast from price history alone."
                if arima_is_whitenoise else
                "captures what limited structure exists, though its forecasts still carry wide confidence bands."
            )
            + f" The GARCH(1,1) model, by contrast, does find clear structure in volatility (persistence = "
            f"{fmt(garch_persistence, 2)}): calm and turbulent periods cluster and are therefore somewhat "
            "forecastable, even though the direction of returns within them is not - a distinction worth "
            "keeping in mind when using this report for risk management rather than return prediction."
        )
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
            "stationarity": stationarity_tbl,
            "autocorr": autocorr_tbl,
            "arima_coef": arima_coef_tbl,
            "arima_forecast": arima_forecast_tbl,
            "arima_diag": arima_diag_tbl,
            "garch_coef": garch_coef_tbl,
            "garch_forecast": garch_forecast_tbl,
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
            "arima_order": (arima_p, 0, arima_q),
            "garch_persistence": garch_persistence,
        },
    }
