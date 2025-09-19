# Easy CFO AI Assistant for Kazakhstan-based startups
# -------------------------
# ~ oriented for Kazakhstan market and beginner startups to make easy financial reports and analytics. We have unique standard for writing unified Excel/CSV reports based on which this program automatically make reports and give you financial insights and recommendations (Groq).

import os
from pathlib import Path
import io
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

if 'GEMINI_API_KEY' in os.environ:
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
else:
    # Если ключа нет, выводим ошибку и завершаем программу
    st.error("Ошибка: Переменная окружения 'GEMINI_API_KEY' не найдена. Пожалуйста, вставьте ваш ключ в файл .env.")
    st.stop()

# Инициализируем модель Gemini
model = genai.GenerativeModel('gemini-1.5-flash-latest')

APP_TITLE = "Easy CFO AI Assistant — Kazakhstan Startup Edition"
BASE_CURRENCIES = ["KZT", "USD", "RUB"] 
DATA_DIR = Path("data")

BAL_PATH = DATA_DIR / "balances.csv"   
FX_PATH = DATA_DIR / "fx.csv"
TODAY = datetime.today().date()

scenario_box = st.selectbox("Scenarios:", ['Scenario 1', 'Scenario 2'], index=0)
if scenario_box == 'Scenario 1':
    PAY_PATH = DATA_DIR / "payments.csv"
else:
    PAY_PATH = DATA_DIR / "payments2.csv" 

# ---------------------------
# Gemini Insights
# ---------------------------

def generate_ai_insights(model, proj: pd.DataFrame, rec: dict, base_ccy: str, horizon_days: int):
    """
    Вызов Google Gemini LLM для генерации аналитики для казахстанских стартапов.
    """
    try:
        summary = proj[["date","closing_cash","planned_in","planned_out"]].tail(14)
        summary_csv = summary.to_csv(index=False)

        user_prompt = f"""
        Вы — финансовый AI-помощник для стартапов в Казахстане.
        Базовая валюта: {base_ccy}.
        Горизонт прогноза: {horizon_days} дней.
        
        Данные 2-недельного прогноза (CSV):
        {summary_csv}

        Ключевые расчетные значения:
        - Минимальный кэш: {rec['min_cash']:.0f} {base_ccy}
        - Дни ниже нуля: {rec['days_below_zero']}
        - Рекомендуемая потребность в кредитной линии: {rec['credit_line_recommendation']:.0f} {base_ccy}
        - Потенциал для депозита: {rec['deposit_recommendation']:.0f} {base_ccy}

        Пожалуйста, предоставьте:
        1. Краткий исполнительный обзор состояния ликвидности.
        2. Риски, специфичные для стартапов в Казахстане (волатильность курса, движение KZT/USD, налоговые сроки, давление на зарплатный фонд).
        3. Действенные рекомендации (денежный буфер, финансирование, инвестиции).
        Сохраняйте текст понятным, очень коротко, практичным и на русском языке.
        """
        
        print(user_prompt)

        # Генерация контента с помощью модели Gemini
        completion = model.generate_content(user_prompt)
        
        return completion.text.strip()
    except Exception as e:
        return f"⚠️ AI-аналитика недоступна: {e}"
    
def ensure_sample_data():
    """Ensures the data directory and sample CSV files exist (Kazakhstan-oriented)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not BAL_PATH.exists():
        bal = pd.DataFrame([
            {"date": TODAY.isoformat(), "account": "Kaspi Main",   "currency": "KZT", "balance": 180_000_000},
            {"date": TODAY.isoformat(), "account": "Halyk Ops",   "currency": "KZT", "balance": 95_000_000},
            {"date": TODAY.isoformat(), "account": "USD Reserve", "currency": "USD", "balance": 250_000},
        ])
        bal.to_csv(BAL_PATH, index=False)

    if not FX_PATH.exists():
        days = pd.date_range(TODAY - timedelta(days=30), TODAY + timedelta(days=60), freq="D")
        fx_rows = []
        for d in days:
            fx_rows += [
                {"date": d.date().isoformat(), "base": "KZT", "quote": "KZT", "rate": 1.0},
                {"date": d.date().isoformat(), "base": "USD", "quote": "KZT", "rate": 480.0},
                {"date": d.date().isoformat(), "base": "RUB", "quote": "KZT", "rate": 5.2},
            ]
        pd.DataFrame(fx_rows).to_csv(FX_PATH, index=False)

    if not PAY_PATH.exists():
        rng_hist = pd.date_range(TODAY - timedelta(days=60), TODAY - timedelta(days=1), freq="D")
        rng_plan = pd.date_range(TODAY, TODAY + timedelta(days=45), freq="D")

        rows = []
        for d in rng_hist:
            # Inflows: mostly sales in KZT
            rows.append({"date": d.date().isoformat(), "amount": abs(np.random.normal(55_000_00, 18_000_00)),
                         "currency": "KZT", "type": "inflow", "category": "Sales",
                         "description": "Kaspi QR / Bank transfers", "status": "actual"})
            # Outflows
            rows.append({"date": d.date().isoformat(), "amount": abs(np.random.normal(40_000_00, 12_000_00)),
                         "currency": "KZT", "type": "outflow", "category": "Suppliers",
                         "description": "Raw materials / Services", "status": "actual"})
            if d.day in (10, 25):  # Payroll twice a month
                rows.append({"date": d.date().isoformat(), "amount": 30_000_00,
                             "currency": "KZT", "type": "outflow", "category": "Payroll",
                             "description": "Employee salaries", "status": "actual"})

        for d in rng_plan:
            if d.weekday() in (0, 2, 4):  # Sales inflows
                rows.append({"date": d.date().isoformat(), "amount": 60_000_00,
                             "currency": "KZT", "type": "inflow", "category": "Sales",
                             "description": "Planned sales inflow", "status": "planned"})
            if d.day in (10, 25):
                rows.append({"date": d.date().isoformat(), "amount": 32_000_00,
                             "currency": "KZT", "type": "outflow", "category": "Payroll",
                             "description": "Planned salaries", "status": "planned"})
            if d.day == 25:  # Taxes monthly
                rows.append({"date": d.date().isoformat(), "amount": 20_000_00,
                             "currency": "KZT", "type": "outflow", "category": "Tax",
                             "description": "VAT / CIT payment", "status": "planned"})

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

    # print("Opening::::", bal)
    # print('len:', len(bal))

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
    st.markdown(
        """
        <a href="https://astanahub.com/l/kak-otkryt-it-kompanyiu-v-kazakhstane" target="_blank">
            <button style="
                background-color:#696969;
                color:white;
                padding:10px 20px;
                border:none;
                border-radius:8px;
                font-size:16px;
                cursor:pointer;
            ">🚀 Join Astana Hub</button>
        </a>
        """,
        unsafe_allow_html=True
    )
    st.title(APP_TITLE)
    st.caption("MVP for hackathon: real-time liquidity view, scenarios, actionable tips.")

    ensure_sample_data()
    bal, pay, fx = load_data()
    if bal is None or pay is None or fx is None:
        return



    # Sidebar settings (same as before) ...
    with st.sidebar:
        st.subheader("Settings")
        base_ccy = st.selectbox("Base currency", BASE_CURRENCIES, index=0)  # default to KZT
        horizon_days = st.slider("Horizon (days)", 7, 60, 30, step=1)
        st.markdown("---")
        st.subheader("Scenarios (What-if)")
        fx_shock_pct = st.slider("FX shock % (non-base)", -20, 20, 0, step=1)
        delay_top_n = st.slider("Delay top-N outflows", 0, 20, 0, step=1)
        delay_days = st.slider("Delay by days", -15, 30, 0, step=1)
        supplier_multiplier = st.slider("Suppliers × multiplier", 0.5, 1.5, 1.0, step=0.05)
        st.markdown("---")
        st.subheader("Policy")
        if not pay[(pay["type"]=="outflow") & (pay["status"]=="actual")].empty:
            avg_outflow = pay[(pay["type"]=="outflow") & (pay["status"]=="actual")]["amount"].mean()
        else:
            avg_outflow = 1.0
        buffer_pct = st.slider("Safety buffer (% of avg daily outflow)", 5, 100, 30, step=5)
        safety_buffer = (buffer_pct/100.0) * avg_outflow

    # Core calculation
    opening, hist_daily, plan_s, proj = build_projection(
        bal, pay, fx, base_ccy, horizon_days,
        fx_shock_pct, delay_top_n, delay_days, supplier_multiplier
    )
    rec = recommendations(proj, opening, safety_buffer)

    # KPIs row...
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Opening (base)", f"{opening:,.0f} {base_ccy}")
    c2.metric("Min cash", f"{rec['min_cash']:,.0f} {base_ccy}")
    c3.metric("Days < 0", rec["days_below_zero"])
    c4.metric("Credit line need", f"{rec['credit_line_recommendation']:,.0f} {base_ccy}")
    c5.metric("Deposit potential", f"{rec['deposit_recommendation']:,.0f} {base_ccy}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Payments", "Download Report", "AI Insights"])

    with tab1:
        st.subheader("Projected closing cash")
        st.line_chart(proj.set_index("date")["closing_cash"])
        st.subheader("Daily components")
        st.bar_chart(proj.set_index("date")[["runrate","planned_in","planned_out"]])
        st.caption("Run-rate is simple EWMA over history. Replace with ML model later.")

    with tab2:
        st.subheader("Upcoming payments (scenario-adjusted)")
        show = plan_s[["date","type","category","amount","currency","amount_base","description"]]
        st.dataframe(show.sort_values(["date","type"], ascending=[True, False]), use_container_width=True)

    with tab3:
        st.subheader("Liquidity table + Recommendations")
        rep = proj.copy()
        rep["opening"] = [opening] + list(proj["closing_cash"].shift(1).fillna(opening))[:-1]
        rep = rep[["date","opening","planned_in","planned_out","runrate","net","closing_cash"]]
        buf = io.StringIO()
        rep.to_csv(buf, index=False)
        st.download_button("Download CSV", buf.getvalue(), file_name="liquidity_projection.csv", mime="text/csv")
        if rec["credit_line_recommendation"] > 0:
            st.write(f"• Credit line needed ≈ **{rec['credit_line_recommendation']:,.0f} {base_ccy}**.")
        else:
            st.write("• No credit line needed under current scenarios.")
        if rec["deposit_recommendation"] > 0:
            st.write(f"• Safe to invest ≈ **{rec['deposit_recommendation']:,.0f} {base_ccy}**.")
        else:
            st.write("• Hold liquidity, avoid deposits now.")

    with tab4:
        st.subheader("🤖 AI Insights (Gemini)")
        # Если клиент Gemini не инициализирован, сообщаем об этом
        # if not genai.get_client().api_key:
        #     st.info("Чтобы получить AI-аналитику, добавьте 'GEMINI_API_KEY' в файл .env.")
        # else:

        if st.button("Create short recommendations"):

            insights = generate_ai_insights(model, proj, rec, base_ccy, horizon_days)
            st.write(insights)

    st.caption("Built for Kazakhstan startups. Not investment advice.")

if __name__ == "__main__":
    main()
