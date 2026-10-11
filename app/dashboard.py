"""RetainIQ Streamlit dashboard.

Run with:  streamlit run app/dashboard.py
Reads the files written by scripts/run_pipeline.py from outputs/real or outputs/sample.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retainiq import config  # noqa: E402

st.set_page_config(page_title="RetainIQ", page_icon="📈", layout="wide")

SEGMENT_ORDER = ["Champions", "Loyal", "New / Promising", "Needs Attention",
                 "At Risk", "Can't Lose", "Hibernating"]


# --------------------------------------------------------------------------- data
def available_output_dirs() -> dict[str, Path]:
    override = os.environ.get("RETAINIQ_OUTPUT_DIR")
    if override:
        path = Path(override)
        return {path.name: path} if (path / "customer_scores.csv").exists() else {}
    found = {}
    for name in ("real", "sample"):
        path = config.OUTPUTS / name
        if (path / "customer_scores.csv").exists():
            found[name] = path
    return found


@st.cache_data(show_spinner=False)
def load_outputs(path: str) -> dict:
    p = Path(path)
    data = {
        "scores": pd.read_csv(p / "customer_scores.csv"),
        "monthly": pd.read_csv(p / "monthly_revenue.csv"),
        "cohort": pd.read_csv(p / "cohort_retention.csv", index_col=0),
        "importance": pd.read_csv(p / "feature_importance.csv"),
        "metrics": json.loads((p / "metrics.json").read_text()),
        "cleaning": json.loads((p / "cleaning_report.json").read_text()),
        "run_info": json.loads((p / "run_info.json").read_text()),
        "sql": {f.stem: pd.read_csv(f) for f in sorted((p / "sql_insights").glob("*.csv"))},
    }
    return data


def money(value: float) -> str:
    return f"£{value:,.0f}"


# --------------------------------------------------------------------------- app
st.title("📈 RetainIQ: Revenue Forecast & Retention Dashboard")

dirs = available_output_dirs()
if not dirs:
    st.error("No pipeline outputs found.")
    st.markdown(
        "Run the pipeline first:\n\n"
        "```bash\npython scripts/run_pipeline.py            # real data in data/raw/\n"
        "python scripts/run_pipeline.py --sample    # synthetic demo data\n```"
    )
    st.stop()

choice = st.sidebar.selectbox("Output set", list(dirs), index=0) if len(dirs) > 1 \
    else list(dirs)[0]
data = load_outputs(str(dirs[choice]))
scores, run_info = data["scores"], data["run_info"]

if run_info["source"] == "synthetic":
    st.warning("**SYNTHETIC DEMO DATA.** Every number below is artificial and is not a "
               "real-world finding. Run the pipeline on the UCI dataset for real results.")

horizon = run_info["horizon_days"]
st.caption(f"Snapshot date: {run_info['snapshot_date'][:10]} · Forecast horizon: "
           f"{horizon} days · Customers scored: {run_info['customers_scored']:,}")

# ----- sidebar filters
st.sidebar.header("Filters")
segments = [s for s in SEGMENT_ORDER if s in set(scores["rfm_segment"])]
sel_segments = st.sidebar.multiselect("RFM segment", segments, default=segments)
sel_tiers = st.sidebar.multiselect("Value tier", ["High", "Mid", "Low"],
                                   default=["High", "Mid", "Low"])
sel_risk = st.sidebar.multiselect("Churn-risk bucket", ["High", "Medium", "Low"],
                                  default=["High", "Medium", "Low"])
countries = sorted(scores["country"].dropna().unique())
sel_countries = st.sidebar.multiselect("Country", countries, default=countries)
min_churn = st.sidebar.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05)

mask = (
    scores["rfm_segment"].isin(sel_segments)
    & scores["value_tier"].isin(sel_tiers)
    & scores["risk_bucket"].isin(sel_risk)
    & scores["country"].isin(sel_countries)
    & (scores["churn_probability"] >= min_churn)
)
view = scores[mask]

# ----- KPIs
total_pred = scores["predicted_revenue"].sum()
at_risk_rev = scores.loc[scores["risk_bucket"] == "High", "predicted_revenue"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers (filtered)", f"{len(view):,}", f"of {len(scores):,}")
c2.metric(f"Predicted revenue, next {horizon}d", money(view["predicted_revenue"].sum()))
c3.metric("Avg churn probability",
          f"{view['churn_probability'].mean():.0%}" if len(view) else "n/a")
c4.metric("Predicted revenue at high risk", money(at_risk_rev),
          f"{at_risk_rev / total_pred:.0%} of total" if total_pred else None)

tabs = st.tabs(["Overview", "Segments & actions", "Customers", "Cohorts",
                "Model performance", "SQL insights"])

with tabs[0]:
    left, right = st.columns(2)
    monthly = data["monthly"]
    fig = px.bar(monthly, x="month", y="revenue", title="Monthly revenue (all customers)")
    left.plotly_chart(fig, use_container_width=True)
    fig = px.line(monthly, x="month", y="customers", markers=True,
                  title="Active customers per month")
    right.plotly_chart(fig, use_container_width=True)
    if len(view):
        fig = px.scatter(view, x="churn_probability", y="predicted_revenue",
                         color="value_tier", hover_data=["CustomerID", "rfm_segment"],
                         title="Predicted revenue vs churn probability",
                         category_orders={"value_tier": ["High", "Mid", "Low"]})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No customers match the current filters.")

with tabs[1]:
    if len(view):
        seg = (view.groupby("rfm_segment")
               .agg(customers=("CustomerID", "count"),
                    predicted_revenue=("predicted_revenue", "sum"),
                    avg_churn=("churn_probability", "mean"))
               .reindex([s for s in SEGMENT_ORDER if s in set(view["rfm_segment"])])
               .reset_index())
        a, b = st.columns(2)
        a.plotly_chart(px.bar(seg, x="rfm_segment", y="customers",
                              title="Customers per RFM segment"), use_container_width=True)
        b.plotly_chart(px.bar(seg, x="rfm_segment", y="predicted_revenue",
                              title="Predicted revenue per RFM segment"),
                       use_container_width=True)
        st.subheader("Recommended retention actions")
        actions = (view.groupby(["value_tier", "risk_bucket", "recommended_action"])
                   .agg(customers=("CustomerID", "count"),
                        predicted_revenue=("predicted_revenue", "sum"))
                   .reset_index()
                   .sort_values("predicted_revenue", ascending=False))
        actions["predicted_revenue"] = actions["predicted_revenue"].round(0)
        st.dataframe(actions, use_container_width=True, hide_index=True)
    else:
        st.info("No customers match the current filters.")

with tabs[2]:
    st.write(f"{len(view):,} customers match the filters.")
    cols = ["CustomerID", "country", "rfm_segment", "value_tier", "risk_bucket",
            "predicted_revenue", "churn_probability", "recency_days", "frequency",
            "monetary", "recommended_action"]
    table = view[cols].sort_values("predicted_revenue", ascending=False)
    st.dataframe(table.round(2), use_container_width=True, hide_index=True)
    st.download_button("Download filtered customers (CSV)",
                       table.to_csv(index=False).encode("utf-8"),
                       file_name="retainiq_filtered_customers.csv", mime="text/csv")

with tabs[3]:
    cohort = data["cohort"]
    if cohort.empty:
        st.info("Not enough data for cohort analysis.")
    else:
        fig = px.imshow(cohort, text_auto=".0%", aspect="auto", color_continuous_scale="Blues",
                        labels={"x": "Months since first purchase", "y": "Acquisition month",
                                "color": "Retained"},
                        title="Cohort retention (share of cohort purchasing in month N)")
        st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    m = data["metrics"]
    st.subheader("Revenue model (evaluated on a later, unseen time window)")
    rev = pd.DataFrame(m["revenue_model"]).T.reset_index().rename(columns={"index": "model"})
    st.dataframe(rev.round(3), use_container_width=True, hide_index=True)
    st.caption("R2 is on raw revenue and is sensitive to a few very large customers; "
               "R2_log_scale is less so. top20pct_revenue_capture = share of actual revenue "
               "earned by the 20% of customers ranked highest by the model.")
    st.subheader("Retention model (will the customer purchase again?)")
    st.json(m["activity_model"])
    imp = data["importance"].head(12)
    st.plotly_chart(px.bar(imp.iloc[::-1], x="importance_mean", y="feature", orientation="h",
                           error_x="importance_std",
                           title="Permutation importance (held-out window)"),
                    use_container_width=True)
    with st.expander("Cleaning report"):
        st.json(data["cleaning"])
    with st.expander("Best hyper-parameters"):
        st.json(m["revenue_best_params"])

with tabs[5]:
    for name, frame in data["sql"].items():
        st.subheader(name.replace("_", " ").title())
        st.dataframe(frame, use_container_width=True, hide_index=True)
