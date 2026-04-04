from __future__ import annotations

import streamlit as st

from main import run_stock_risk_agent
from ui.chart_service import get_chart_time_series


def _format_currency(value):
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def _format_pct(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def main():
    st.set_page_config(page_title="Stock Risk Agent", layout="wide")
    st.title("Stock Risk Agent")

    with st.sidebar:
        ticker = st.text_input("Ticker", value="TSLA").strip().upper()
        benchmark = st.text_input("Benchmark", value="SPY").strip().upper()
        lookback = st.number_input("Risk Lookback", min_value=30, max_value=2000, value=252, step=1)
        use_real_market = st.checkbox("Use Real Market Data", value=True)
        disable_llm = st.checkbox("Disable LLM Summary", value=False)
        run = st.button("Run Analysis", type="primary")

    if not run:
        st.info("Select options and click Run Analysis.")
        return

    if not ticker:
        st.error("Ticker is required.")
        return

    output, debug = run_stock_risk_agent(
        ticker=ticker,
        benchmark=benchmark or "SPY",
        lookback=int(lookback),
        disable_llm=disable_llm,
        debug=False,
        prefer_real_market_data=use_real_market,
    )

    stat_col, struct_col, src_col = st.columns(3)
    stat_col.metric("Statistical Risk", f"{output['statistical_risk']['score']:.2f}", output["statistical_risk"]["level"])
    struct_col.metric("Structural Risk", f"{output['structural_risk']['score']:.2f}", output["structural_risk"]["level"])
    src_col.write("**Data Sources**")
    src_col.write(f"Market: {debug['data_source'].get('market_data_source')}")
    src_col.write(f"Structural: {debug['data_source'].get('structural_data_source')}")

    chart_data = get_chart_time_series(ticker=ticker, lookback_days=30)

    st.subheader(chart_data["title"])
    if not chart_data["success"]:
        st.warning(chart_data["error"])
        return

    stats = chart_data["stats"] or {}
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Latest Price", _format_currency(stats.get("latest_price")))
    p2.metric("30-Day High", _format_currency(stats.get("high_30d")))
    p3.metric("30-Day Low", _format_currency(stats.get("low_30d")))
    p4.metric("30-Day Return", _format_pct(stats.get("return_30d_pct")))

    points = chart_data["points"]
    st.line_chart(
        data=[{"date": row["date"], "price": row["price"]} for row in points],
        x="date",
        y="price",
    )


if __name__ == "__main__":
    main()