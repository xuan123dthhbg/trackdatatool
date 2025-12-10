import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Retention interpolation
# -----------------------------
def build_retention_curve(
    d1: float,
    d7: float,
    d14: float,
    d40: float,
    d45: float,
    horizon_days: int = 180,
) -> np.ndarray:
    """
    Return retention array r[0..horizon_days], as fraction (0-1).
    r[0] = 1.0 (day of install)
    D1..D45 are given in percentage.
    We interpolate in log-space between anchor points.
    """
    # convert to fraction and avoid zeros for log
    eps = 1e-9
    anchors_days = [0, 1, 7, 14, 40, 45]
    anchors_vals = [
        1.0,
        max(d1 / 100.0, eps),
        max(d7 / 100.0, eps),
        max(d14 / 100.0, eps),
        max(d40 / 100.0, eps),
        max(d45 / 100.0, eps),
    ]

    log_anchors_vals = [math.log(v) for v in anchors_vals]

    r = np.zeros(horizon_days + 1, dtype=float)

    for day in range(horizon_days + 1):
        if day <= anchors_days[0]:
            r[day] = anchors_vals[0]
            continue

        # find interval
        if day >= anchors_days[-1]:
            # beyond last anchor: continue with last segment slope
            d_prev = anchors_days[-2]
            d_next = anchors_days[-1]
            log_prev = log_anchors_vals[-2]
            log_next = log_anchors_vals[-1]
        else:
            # between anchors
            for i in range(1, len(anchors_days)):
                if day <= anchors_days[i]:
                    d_prev = anchors_days[i - 1]
                    d_next = anchors_days[i]
                    log_prev = log_anchors_vals[i - 1]
                    log_next = log_anchors_vals[i]
                    break

        if d_next == d_prev:
            log_val = log_next
        else:
            t = (day - d_prev) / (d_next - d_prev)
            log_val = log_prev + t * (log_next - log_prev)

        val = math.exp(log_val)
        r[day] = min(max(val, 0.0), 1.0)  # clamp 0-1

    return r


def simulate_ua(
    cost_per_day: float,
    cpi: float,
    arpdau: float,
    ua_days: int,
    horizon_days: int,
    retention_curve: np.ndarray,
) -> pd.DataFrame:
    """
    Simulate UA running cost_per_day for ua_days, with retention_curve[0..H].
    """
    H = horizon_days
    # daily cost
    cost = np.zeros(H, dtype=float)
    cost[: min(ua_days, H)] = cost_per_day

    # installs per day
    installs = np.where(cost > 0, cost / cpi if cpi > 0 else 0.0, 0.0)

    # DAU via convolution: DAU[t] = sum_k installs[k] * r[t-k]
    # retention_curve length must be >= H
    r = retention_curve[: H]  # r[0..H-1] enough for convolution
    dau_full = np.convolve(installs, r)[:H]

    # revenue
    revenue = dau_full * arpdau

    cum_cost = np.cumsum(cost)
    cum_revenue = np.cumsum(revenue)
    roas = np.where(cum_cost > 0, cum_revenue / cum_cost, np.nan)

    days = np.arange(1, H + 1)

    df = pd.DataFrame(
        {
            "Day": days,
            "Cost_per_day": cost,
            "Installs": installs,
            "DAU": dau_full,
            "Revenue": revenue,
            "Cum_Cost": cum_cost,
            "Cum_Revenue": cum_revenue,
            "ROAS_cum": roas,
        }
    )
    return df


def find_payback_day(df: pd.DataFrame) -> int:
    cond = df["Cum_Revenue"] >= df["Cum_Cost"]
    idx = np.where(cond.values)[0]
    if len(idx) == 0:
        return -1
    return int(df.loc[idx[0], "Day"])


# -----------------------------
# Streamlit app
# -----------------------------
def main():
    st.set_page_config(
        page_title="UA Payback Simulator",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 UA Payback Simulator – Retention & LTV")
    st.caption(
        "Nhập vài mốc Retention + Cost per day, CPI, ARPDAU → mô phỏng DAU, doanh thu và thời điểm hồi vốn."
    )

    with st.sidebar:
        st.header("⚙️ Cấu hình mô phỏng")
        horizon_days = st.number_input(
            "Horizon (số ngày mô phỏng)",
            min_value=30,
            max_value=365,
            value=180,
            step=10,
        )
        ua_days = st.number_input(
            "Số ngày chạy UA (spend liên tục)",
            min_value=1,
            max_value=horizon_days,
            value=30,
        )

        st.markdown("---")
        st.markdown("### 💵 Thông số UA & Monetization")
        cost_per_day = st.number_input(
            "Cost per day (ngân sách/ngày)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
        )
        cpi = st.number_input(
            "CPI dự kiến",
            min_value=0.01,
            value=1.5,
            step=0.05,
        )
        arpdau = st.number_input(
            "ARPDAU (doanh thu/DAU/ngày)",
            min_value=0.0,
            value=0.25,
            step=0.01,
        )

        st.markdown("---")
        st.markdown("### 📉 Retention Inputs (%)")
        d1 = st.number_input("D1 retention (%)", min_value=0.0, max_value=100.0, value=35.0, step=0.5)
        d7 = st.number_input("D7 retention (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.5)
        d14 = st.number_input("D14 retention (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        d40 = st.number_input("D40 retention (%)", min_value=0.0, max_value=100.0, value=8.0, step=0.5)
        d45 = st.number_input("D45 retention (%)", min_value=0.0, max_value=100.0, value=7.0, step=0.5)

        st.markdown(
            "<small>Retention tính theo active user/installs, D1 là ngày sau install 1 ngày, v.v.</small>",
            unsafe_allow_html=True,
        )

    # Core simulation
    retention_curve = build_retention_curve(d1, d7, d14, d40, d45, horizon_days=horizon_days)
    df = simulate_ua(cost_per_day, cpi, arpdau, ua_days, horizon_days, retention_curve)
    payback_day = find_payback_day(df)

    # Top KPIs
    total_installs = df["Installs"].sum()
    total_cost = df["Cum_Cost"].iloc[-1]
    total_revenue = df["Cum_Revenue"].iloc[-1]
    final_roas = total_revenue / total_cost if total_cost > 0 else float("nan")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Installs", f"{total_installs:,.0f}")
    col2.metric("Total Cost", f"{total_cost:,.0f}")
    col3.metric("Total Revenue", f"{total_revenue:,.0f}")
    col4.metric("ROAS (cum)", f"{final_roas:.2f}x" if total_cost > 0 else "N/A")

    col5, col6 = st.columns(2)
    if payback_day > 0:
        col5.metric("Payback Day", f"Day {payback_day}")
        payback_period = payback_day  # days since start
        col6.metric("Thời gian hồi vốn", f"{payback_period} ngày")
    else:
        col5.metric("Payback Day", "Chưa hồi vốn trong horizon")
        col6.metric("Thời gian hồi vốn", "-")

    # Layout charts
    tab1, tab2, tab3 = st.tabs(
        [
            "Retention curve",
            "Daily metrics (DAU & Revenue)",
            "Cumulative Cost vs Revenue",
        ]
    )

    with tab1:
        st.subheader("📉 Retention Curve (0–Horizon)")
        days = np.arange(0, horizon_days + 1)
        fig_ret = go.Figure()
        fig_ret.add_trace(
            go.Scatter(
                x=days,
                y=retention_curve * 100.0,
                mode="lines",
                name="Retention (fitted)",
            )
        )
        fig_ret.add_trace(
            go.Scatter(
                x=[1, 7, 14, 40, 45],
                y=[d1, d7, d14, d40, d45],
                mode="markers+text",
                name="Anchors",
                text=[f"D{x}" for x in [1, 7, 14, 40, 45]],
                textposition="top center",
            )
        )
        fig_ret.update_layout(
            xaxis_title="Day",
            yaxis_title="Retention (%)",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    with tab2:
        st.subheader("📊 DAU & Revenue theo ngày")

        fig_daily = go.Figure()
        fig_daily.add_trace(
            go.Scatter(
                x=df["Day"],
                y=df["DAU"],
                mode="lines",
                name="DAU",
            )
        )
        fig_daily.add_trace(
            go.Scatter(
                x=df["Day"],
                y=df["Revenue"],
                mode="lines",
                name="Revenue / day",
                yaxis="y2",
            )
        )

        fig_daily.update_layout(
            xaxis_title="Day",
            yaxis=dict(title="DAU", rangemode="tozero"),
            yaxis2=dict(
                title="Revenue",
                overlaying="y",
                side="right",
                rangemode="tozero",
            ),
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        st.plotly_chart(fig_daily, use_container_width=True)

        st.markdown("#### Bảng tóm tắt theo ngày (top 60 ngày đầu)")
        st.dataframe(
            df.loc[:59, ["Day", "Installs", "DAU", "Revenue", "Cum_Cost", "Cum_Revenue", "ROAS_cum"]],
            use_container_width=True,
        )

    with tab3:
        st.subheader("💰 Cumulative Cost vs Revenue")

        fig_cum = go.Figure()
        fig_cum.add_trace(
            go.Scatter(
                x=df["Day"],
                y=df["Cum_Cost"],
                mode="lines",
                name="Cumulative Cost",
            )
        )
        fig_cum.add_trace(
            go.Scatter(
                x=df["Day"],
                y=df["Cum_Revenue"],
                mode="lines",
                name="Cumulative Revenue",
            )
        )

        if payback_day > 0:
            payback_row = df[df["Day"] == payback_day].iloc[0]
            fig_cum.add_trace(
                go.Scatter(
                    x=[payback_day],
                    y=[payback_row["Cum_Revenue"]],
                    mode="markers+text",
                    name="Payback point",
                    text=[f"Day {payback_day}"],
                    textposition="bottom right",
                )
            )

        fig_cum.update_layout(
            xaxis_title="Day",
            yaxis_title="Amount",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
        )

        st.plotly_chart(fig_cum, use_container_width=True)

        st.markdown("#### Snapshot một vài mốc chính")
        snapshot_days = [30, 60, 90, 120, horizon_days]
        snapshot_rows = df[df["Day"].isin(snapshot_days)]
        st.dataframe(
            snapshot_rows[["Day", "Cum_Cost", "Cum_Revenue", "ROAS_cum"]],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
