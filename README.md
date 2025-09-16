# Easy CFO AI Assistant  
**For Kazakhstan-based Startups**

**Easy CFO AI Assistant** is a lightweight financial analytics tool built for **early-stage startups in Kazakhstan**.  
It helps you **unify your financial data (Excel/CSV)**, generate **automated reports**, and receive **AI-powered insights & recommendations**.

The app expects **3 standard CSV files** (or you can use auto-generated demo data):

- `balances.csv` → account balances (Kaspi, Halyk, FX reserves)  
- `fx.csv` → currency exchange rates (base `KZT`)  
- `payments.csv` → inflows/outflows (Sales, Suppliers, Payroll, Tax, etc.)  

This creates a **simple but powerful unified reporting system**.

## Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/DearZiZiZi/shai.pro
   cd shai.pro
   ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

> Don't forget about creating `.env` file with API key from [Groq](https://groq.com/)

3. **Run the app**
    ```
    streamlit run app.py
    ```

TO-DO:
1. Study specific financial metrics and features of the Kazakhstan market for tech startups.
2. Apply classic ML algorithms to make base forecasts.
3. Integrate [Shai.pro](http://shai.pro/) AI agent to make advanced research & recommendation based on the startup's financial report.