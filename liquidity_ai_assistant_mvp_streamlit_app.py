# Liquidity AI Assistant — MVP (Streamlit)
# -------------------------------------------------------------
# One-file prototype you can run in ~minutes.
# - Generates sample data if none present (./data/*.csv)
# - Aggregates multi-currency balances & payments
# - Projects 30–60 day liquidity with a simple EWMA run-rate + planned cash flows
# - What‑if scenarios: FX shock, delay top outflows, change supplier spend
# - Recommendations: credit line need, deposit amount
# -------------------------------------------------------------

import os
from pathlib import Path
import io
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import streamlit as st

APP_TITLE = "Financial AI Assistant — Liquidity MVP"
BASE_CURRENCIES = ["USD", "EUR", "KZT", "UZS"]
DATA_DIR = Path("data")
# Correcting file paths to be relative to the script's directory
BAL_PATH = DATA_DIR / "citadel_highsens_balances.csv"
PAY_PATH = DATA_DIR / "citadel_highsens_payments.csv"
FX_PATH = DATA_DIR / "citadel_highsens_fx_rates.csv"
TODAY = datetime.today().date()

# ---------------------------
# Utilities & sample data
# ---------------------------

def ensure_sample_data():
    """Ensures the data directory and sample CSV files exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not BAL_PATH.exists():
        # Opening balances per account (multi-currency)
        bal = pd.DataFrame([
            {"date": TODAY.isoformat(), "account": "Main USD", "currency": "USD", "balance": 850000},
            {"date": TODAY.isoformat(), "account": "Ops EUR",  "currency": "EUR", "balance": 210000},
            {"date": TODAY.isoformat(), "account": "Sales KZT", "currency": "KZT", "balance": 380_000_000},
            {"date": TODAY.isoformat(), "account": "Reserve UZS","currency": "UZS", "balance": 5_400_000_000},
        ])
        bal.to_csv(BAL_PATH, index=False)

    if not FX_PATH.exists():
        # Daily FX to USD (very rough sample, for demo only)
        days = pd.date_range(TODAY - timedelta(days=30), TODAY + timedelta(days=60), freq="D")
        fx_rows = []
        for d in days:
            fx_rows += [
                {"date": d.date().isoformat(), "base": "USD", "quote": "USD", "rate": 1.0},
                {"date": d.date().isoformat(), "base": "EUR", "quote": "USD", "rate": 1.08},
                {"date": d.date().isoformat(), "base": "KZT", "quote": "USD", "rate": 0.0021},
                {"date": d.date().isoformat(), "base": "UZS", "quote": "USD", "rate": 0.000079},
            ]
        pd.DataFrame(fx_rows).to_csv(FX_PATH, index=False)

    if not PAY_PATH.exists():
        # Generate synthetic history (last 60d) + plan (next 45d)
        rng_hist = pd.date_range(TODAY - timedelta(days=60), TODAY - timedelta(days=1), freq="D")
        rng_plan = pd.date_range(TODAY, TODAY + timedelta(days=45), freq="D")

        rows = []
        # History: random-ish sales (inflows) & ops costs (outflows)
        for d in rng_hist:
            # Inflows (USD/KZT)
            rows.append({"date": d.date().isoformat(), "amount": abs(np.random.normal(120000, 35000)),
                         "currency": np.random.choice(["USD", "KZT"], p=[0.6, 0.4]),
                         "type": "inflow", "category": "Sales", "description": "Customer receipts", "status": "actual"})
            # Outflows
            rows.append({"date": d.date().isoformat(), "amount": abs(np.random.normal(90000, 25000)),
                         "currency": np.random.choice(["USD", "EUR", "KZT"], p=[0.4, 0.2, 0.4]),
                         "type": "outflow", "category": "Suppliers", "description": "Pay suppliers", "status": "actual"})
            if d.day in (5, 20):
                rows.append({"date": d.date().isoformat(), "amount": 130000,
                             "currency": "USD", "type": "outflow", "category": "Payroll",
                             "description": "Payroll", "status": "actual"})

        # Plan: deterministic upcoming payments
        for d in rng_plan:
            if d.weekday() in (0, 2, 4):  # Mon/Wed/Fri sales inflows
                rows.append({"date": d.date().isoformat(), "amount": 140000,
                             "currency": "USD", "type": "inflow", "category": "Sales",
                             "description": "Planned sales", "status": "planned"})
            if d.day in (5, 20):
                rows.append({"date": d.date().isoformat(), "amount": 140000,
                             "currency": "USD", "type": "outflow", "category": "Payroll",
                             "description": "Planned payroll", "status": "planned"})
            if d.weekday() == 3:  # Thu suppliers batch
                rows.append({"date": d.date().isoformat(), "amount": 110000,
                             "currency": np.random.choice(["USD", "EUR", "KZT"], p=[0.5, 0.2, 0.3]),
                             "type": "outflow", "category": "Suppliers",
                             "description": "Planned suppliers", "status": "planned"})
            if d.day == 28:
                rows.append({"date": d.date().isoformat(), "amount": 90000,
                             "currency": "USD", "type": "outflow", "category": "Tax",
                             "description": "Tax payment", "status": "planned"})

        pd.DataFrame(rows).to_csv(PAY_PATH, index=False)


def load_data():
    """Loads and normalizes data from CSV files."""
    try:
        bal = pd.read_csv(BAL_PATH)
        pay = pd.read_csv(PAY_PATH)
        fx  = pd.read_csv(FX_PATH)

        # normalize types
        bal["date"] = pd.to_datetime(bal["date"]).dt.date
        pay["date"] = pd.to_datetime(pay["date"]).dt.date
        fx["date"]  = pd.to_datetime(fx["date"]).dt.date
        
        return bal, pay, fx
    except FileNotFoundError:
        st.error("Ошибка: Файлы данных не найдены. Убедитесь, что они созданы в папке 'data'.")
        return None, None, None

def get_fx_series(fx: pd.DataFrame, base: str, to_quote: str):
    if base == to_quote:
        # identity rate = 1
        days = sorted(fx["date"].unique())
        return pd.Series(1.0, index=days)
    df = fx[(fx["base"] == base) & (fx["quote"] == to_quote)].copy()
    if df.empty:
        # Try to invert if available
        inv = fx[(fx["base"] == to_quote) & (fx["quote"] == base)].copy()
        if inv.empty:
            # Fallback constant 1 for unknown pairs
            days = sorted(fx["date"].unique())
            return pd.Series(1.0, index=days)
        inv["rate"] = 1.0 / inv["rate"].replace(0, np.nan)
        inv = inv.dropna(subset=["rate"]).copy()
        df = inv
    df = df.sort_values("date")
    s = pd.Series(df["rate"].values, index=df["date"].values)
    return s


def convert_amount_on_date(amount: float, ccy: str, date: datetime.date, fx: pd.DataFrame, to_quote: str,
                           fx_shock_pct: float = 0.0):
    series = get_fx_series(fx, ccy, to_quote)
    rate = series.get(date, series.iloc[-1] if not series.empty else 1.0)
    rate = rate * (1.0 + fx_shock_pct/100.0) if (ccy != to_quote and fx_shock_pct != 0) else rate
    return amount * rate


# ---------------------------
# Simple forecasting (EWMA run-rate)
# ---------------------------

def ewma_forecast(series: pd.Series, horizon: int, alpha: float = 0.3):
    """Very simple EWMA; returns a constant forecast equal to last EWMA value."""
    if series.empty:
        return pd.Series([0.0] * horizon)
    ewma = series.ewm(alpha=alpha).mean().iloc[-1]
    return pd.Series([float(ewma)] * horizon)


# ---------------------------
# Scenario engine
# ---------------------------

def apply_scenarios(plan_df: pd.DataFrame, base_ccy: str, fx: pd.DataFrame,
                    fx_shock_pct: float, delay_top_n: int, delay_days: int,
                    supplier_multiplier: float):
    df = plan_df.copy()

    # FX shock: apply only to non-base planned items by converting with shocked rate at execution date
    if fx_shock_pct != 0:
        df["amount_base"] = df.apply(lambda r: convert_amount_on_date(
            r["amount"], r["currency"], r["date"], fx, base_ccy, fx_shock_pct), axis=1)
    else:
        df["amount_base"] = df["amount_base"]  # no change (already computed)

    # Delay top-N outflows by D days (by absolute amount in base)
    if delay_top_n > 0 and delay_days != 0:
        outflows = df[df["type"] == "outflow"].copy()
        top_idx = outflows.nlargest(delay_top_n, "amount_base").index
        df.loc[top_idx, "date"] = df.loc[top_idx, "date"] + timedelta(days=delay_days)

    # Supplier multiplier
    if supplier_multiplier != 1.0:
        mask = (df["category"].str.lower() == "suppliers")
        df.loc[mask, "amount_base"] *= supplier_multiplier

    return df


# ---------------------------
# Core projection
# ---------------------------

def build_projection(bal: pd.DataFrame, pay: pd.DataFrame, fx: pd.DataFrame,
                     base_ccy: str, horizon_days: int,
                     fx_shock_pct: float, delay_top_n: int, delay_days: int,
                     supplier_multiplier: float):

    # Opening balance (sum of all accounts converted to base)
    opening = 0.0
    for _, r in bal.iterrows():
        opening += convert_amount_on_date(r["balance"], r["currency"], r["date"], fx, base_ccy)

    # Split history vs plan
    hist = pay[pay["status"] == "actual"].copy()
    plan = pay[pay["status"] == "planned"].copy()

    # Convert to base for history & plan
    for df in (hist, plan):
        df["amount_base"] = df.apply(lambda r: convert_amount_on_date(
            r["amount"], r["currency"], r["date"], fx, base_ccy), axis=1)

    # Daily net history (in base)
    hist_daily = hist.groupby("date").apply(lambda g: g.apply(lambda r: r["amount_base"] if r["type"] == "inflow" else -r["amount_base"], axis=1).sum())
    hist_daily = hist_daily.rename("net").sort_index()

    # Forecast run-rate for horizon
    horizon = pd.date_range(TODAY, TODAY + timedelta(days=horizon_days-1), freq="D").date
    runrate = ewma_forecast(hist_daily, horizon_days, alpha=0.35)
    runrate.index = horizon

    # Apply scenarios to plan
    plan = apply_scenarios(plan, base_ccy, fx, fx_shock_pct, delay_top_n, delay_days, supplier_multiplier)

    # Planned inflows/outflows per day
    plan_in = plan[plan["type"] == "inflow"].groupby("date")["amount_base"].sum().reindex(horizon, fill_value=0.0)
    plan_out = plan[plan["type"] == "outflow"].groupby("date")["amount_base"].sum().reindex(horizon, fill_value=0.0)

    df = pd.DataFrame({
        "date": horizon,
        "runrate": runrate.values,
        "planned_in": plan_in.values,
        "planned_out": plan_out.values,
    })
    df["net"] = df["runrate"] + df["planned_in"] - df["planned_out"]

    # Compute cash trajectory
    cash = []
    bal = opening
    for v in df["net"].values:
        cash.append(bal + v)
        bal = bal + v
    df["closing_cash"] = cash

    return opening, hist_daily, plan, df


# ---------------------------
# Recommendation rules
# ---------------------------

def recommendations(df: pd.DataFrame, opening: float, safety_buffer: float):
    min_cash = float(df["closing_cash"].min()) if not df.empty else opening
    days_below_zero = int((df["closing_cash"] < 0).sum()) if not df.empty else 0

    credit_need = max(0.0, -(min_cash - safety_buffer)) if min_cash < safety_buffer else 0.0

    # Available to invest: peak cash - buffer, but not if any day < buffer
    peak_cash = float(df["closing_cash"].max()) if not df.empty else opening
    can_invest = 0.0
    if (df["closing_cash"] >= safety_buffer).all():
        can_invest = max(0.0, peak_cash - safety_buffer)

    return {
        "min_cash": min_cash,
        "days_below_zero": days_below_zero,
        "credit_line_recommendation": credit_need,
        "deposit_recommendation": can_invest,
    }


# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("MVP for hackathon: real‑time liquidity view, scenarios, actionable tips.")

    ensure_sample_data()
    
    bal, pay, fx = load_data()
    if bal is None or pay is None or fx is None:
        return # Exit if data loading fails

    with st.sidebar:
        st.subheader("Settings")
        base_ccy = st.selectbox("Base currency", BASE_CURRENCIES, index=0)
        horizon_days = st.slider("Horizon (days)", 7, 60, 30, step=1)
        st.markdown("---")
        st.subheader("Scenarios (What‑if)")
        fx_shock_pct = st.slider("FX shock % (non‑base)", -20, 20, 0, step=1)
        delay_top_n = st.slider("Delay top‑N outflows", 0, 20, 0, step=1)
        delay_days = st.slider("Delay by days", -15, 30, 0, step=1)
        supplier_multiplier = st.slider("Suppliers × multiplier", 0.5, 1.5, 1.0, step=0.05)
        st.markdown("---")
        st.subheader("Policy")
        # Added a check to prevent errors on empty dataframes
        if not pay[(pay["type"] == "outflow") & (pay["status"] == "actual")].empty:
            avg_outflow = pay[(pay["type"]=="outflow") & (pay["status"]=="actual")]["amount"].mean()
        else:
            avg_outflow = 1.0
        
        buffer_pct = st.slider("Safety buffer (% of avg daily outflow)", 5, 100, 30, step=5)
        safety_buffer = (buffer_pct/100.0) * avg_outflow

    opening, hist_daily, plan_s, proj = build_projection(
        bal, pay, fx, base_ccy, horizon_days, fx_shock_pct, delay_top_n, delay_days, supplier_multiplier
    )

    rec = recommendations(proj, opening, safety_buffer)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Opening (base)", f"{opening:,.0f} {base_ccy}")
    c2.metric("Min cash", f"{rec['min_cash']:,.0f} {base_ccy}")
    c3.metric("Days < 0", rec["days_below_zero"])
    c4.metric("Credit line need", f"{rec['credit_line_recommendation']:,.0f} {base_ccy}")
    c5.metric("Deposit potential", f"{rec['deposit_recommendation']:,.0f} {base_ccy}")

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Payments", "Download Report"])

    with tab1:
        st.subheader("Projected closing cash")
        st.line_chart(proj.set_index("date")["closing_cash"])  # Streamlit quick chart

        st.subheader("Daily components")
        st.bar_chart(proj.set_index("date")[["runrate","planned_in","planned_out"]])

        st.caption("Run‑rate is simple EWMA over historical daily net. Replace with your ML model later.")

    with tab2:
        st.subheader("Upcoming payments (planned, scenario‑adjusted)")
        show = plan_s.copy()[["date","type","category","amount","currency","amount_base","description"]]
        show = show.sort_values(["date","type"], ascending=[True, False]).reset_index(drop=True)
        st.dataframe(show, use_container_width=True)

    with tab3:
        st.subheader("Liquidity table")
        rep = proj.copy()
        rep["opening"] = [opening] + list(proj["closing_cash"].shift(1).fillna(opening))[:-1]
        rep = rep[["date","opening","planned_in","planned_out","runrate","net","closing_cash"]]

        # CSV buffer
        buf = io.StringIO()
        rep.to_csv(buf, index=False)
        st.download_button("Download CSV", buf.getvalue(), file_name="liquidity_projection.csv", mime="text/csv")

        # Text recommendations
        st.subheader("Recommendations")
        if rec["credit_line_recommendation"] > 0:
            st.write(f"• Consider opening/using a short‑term credit line ≈ **{rec['credit_line_recommendation']:,.0f} {base_ccy}** to stay above buffer.")
        else:
            st.write("• No credit line needed under current scenarios (above buffer).")

        if rec["deposit_recommendation"] > 0:
            st.write(f"• You can safely allocate ≈ **{rec['deposit_recommendation']:,.0f} {base_ccy}** to a short‑term deposit/instrument (tenor < earliest potential shortfall).")
        else:
            st.write("• Hold liquidity; avoid locking funds into term deposits under current scenarios.")

    with st.expander("📎 Data locations & schema"):
        st.markdown(
            f"""
            **CSV files (auto‑generated on first run):** `./data/\*`

            - `bank_balances.csv`: `date, account, currency, balance`
            - `payments.csv`: `date, amount, currency, type[inflow|outflow], category, description, status[actual|planned]`
            - `fx_rates.csv`: `date, base, quote, rate` (used for conversion to base currency)

            Change/replace these with real exports (bank API, ERP, Treasury).
            """
        )

    st.caption("Built for hackathon demo. Not investment advice. Replace sample FX & EWMA with real sources/models.")

if __name__ == "__main__":
    main()