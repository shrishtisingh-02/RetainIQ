"""Central configuration. All paths are relative to the project root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
SQL_FILE = ROOT / "sql" / "queries.sql"

# Raw file names the loader looks for inside data/raw/ (first match wins).
RAW_CANDIDATES = [
    "Online Retail.xlsx",
    "Online_Retail.xlsx",
    "online_retail.xlsx",
    "Online Retail.csv",
    "online_retail.csv",
    "Raw_Data.csv",
]

REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID",
]
OPTIONAL_COLUMNS = ["Description", "Country"]

# Alternative column names (e.g. the "Online Retail II" file) mapped to the canonical ones.
COLUMN_ALIASES = {
    "invoice": "InvoiceNo",
    "invoiceno": "InvoiceNo",
    "stockcode": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoicedate": "InvoiceDate",
    "price": "UnitPrice",
    "unitprice": "UnitPrice",
    "customerid": "CustomerID",
    "customer id": "CustomerID",
    "country": "Country",
}

# Stock codes that are fees / adjustments rather than products.
NON_PRODUCT_CODES = {
    "POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK", "C2",
    "AMAZONFEE", "S", "B", "ADJUST", "ADJUST2", "TEST001", "TEST002",
}

# Forecast horizon: how far ahead revenue / churn are predicted.
# 90 days is the default because the UCI dataset spans only ~13 months, and a
# time-based train/test split needs two full horizons plus some history.
DEFAULT_HORIZON_DAYS = 90
MIN_HISTORY_DAYS = 60   # minimum history before the earliest training cutoff
MIN_CUSTOMERS = 50      # minimum customers in a training table for CV to make sense

RANDOM_STATE = 42

# Retention decision thresholds
HIGH_VALUE_TOP_PCT = 0.20     # top 20% by predicted revenue -> "High"
MID_VALUE_TOP_PCT = 0.50      # next 30% -> "Mid", rest -> "Low"
HIGH_RISK_THRESHOLD = 0.60    # churn probability >= 0.60 -> "High" risk
MEDIUM_RISK_THRESHOLD = 0.30  # 0.30 - 0.60 -> "Medium", below -> "Low"
