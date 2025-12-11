import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================
# Retention curve (log-linear)
# Anchors: D0, D1, D3, D7, D14, D30, D45, D60, D90, D120
# =============================
def build_retention_curve(
    d1: float, d3: float, d7: float, d14: float, d30: float,
    d45: float, d60: float, d90: float, d120: float,
    horizon_days: int = 180,
    enforce_monotonic: bool = True,
) -> np.ndarray:
    eps = 1e-9
    anchors_days = np.array([0, 1, 3, 7, 14, 30, 45, 60, 90, 120], dtype=int)
    anchors_vals = np.array(
        [
            1.0,
            max(d1 / 100.0, eps),
            max(d3 / 100.0, eps),
            max(d7 / 100.0, eps),
            max(d14 / 100.0, eps),
            max(d30 / 100.0, eps),
            max(d45 / 100.0, eps),
            max(d60 / 100.0, eps),
            max(d90 / 100.0, eps),
            max(d120 / 100.0, eps),
        ],
        dtype=float,
    )

    if enforce_monotonic:
        for i in range(1, len(anchors_vals)):
            anchors_vals[i] = min(anchors_vals[i], anchors_vals[i - 1])

    log_anchors = np.log(np.maximum(anchors_vals, eps))
    r = np.zeros(horizon_days + 1, dtype=float)

    for day in range(horizon_days + 1):
        if day <= anchors_days[0]:
            r[day] = anchors_vals[0]
            continue

        if day >= anchors_days[-1]:
            # extend using last segment slope (90->120)
            d_prev, d_next = anchors_days[-2], anchors_days[-1]
            log_prev, log_next = log_anchors[-2], log_anchors[-1]
        else:
            idx = np.searchsorted(anchors_days, day, side="left")
            d_prev, d_next = anchors_days[idx - 1], anchors_days[idx]
            log_prev, log_next = log_anchors[idx - 1], log_anchors[idx]

        t = 0.0 if d_next == d_prev else (day - d_prev) / (d_next - d_prev)
        log_val = log_prev + t * (log_next - log_prev)
        r[day] = float(np.clip(np.exp(log_val), 0.0, 1.0))

    return r


# =============================
# UA simulation (cohort convolution)
# - Cost only UA cost (cost/day)
# - Revenue net = gross - tracking_cost(% of gross)
# =============================
def simulate_ua(
    cost_per_day: float,
    cpi: float,
    arpdau: float,
    ua_days: int,
    horizon_days: int,
    retention_curve: np.ndarray,
    tracking_pct: float = 0.025,
) -> pd.DataFrame:
    H = int(horizon_days)
    ua_days = int(min(max(ua_days, 0), H))

    cost = np.zeros(H, dtype=float)
    cost[:ua_days] = float(cost_per_day)

    installs = np.where(cost > 0, cost / float(cpi), 0.0)

    # retention_curve includes r[0]=1
    r = retention_curve[:H]
    dau = np.convolve(installs, r)[:H]

    revenue_gross = dau * float(arpdau)
    tracking_cost = revenue_gross * float(tracking_pct)
    revenue_net = revenue_gross - tracking_cost

    daily_pnl = revenue_net - cost

    cum_cost = np.cumsum(cost)
    cum_rev = np.cumsum(revenue_net)
    cum_pnl = np.cumsum(daily_pnl)

    roas_cum = np.where(cum_cost > 0, cum_rev / cum_cost, np.nan)

    df = pd.DataFrame(
        {
            "Day": np.arange(1, H + 1),
            "Cost_UA": cost,
            "NIU": installs,
            "DAU": dau,
            "Revenue_Gross": revenue_gross,
            "TrackingCost": tracking_cost,
            "Revenue_Net": revenue_net,
            "Daily_PnL": daily_pnl,
            "Cum_Cost": cum_cost,
            "Cum_Revenue": cum_rev,
            "Cum_PnL": cum_pnl,
            "ROAS_cum": roas_cum,
        }
    )
    return df


def find_payback_day(df: pd.DataFrame) -> int:
    hit = np.where((df["Cum_Revenue"] >= df["Cum_Cost"]).values)[0]
    return int(df.loc[hit[0], "Day"]) if len(hit) else -1


# =============================
# LTV calculations
# 2 methods:
# - "Excel weights": Lifespan = R1*2.5 + R7*7 + R14*12 + R30*57.5 + R180*100
# - "Curve sum": Lifespan = sum_{day=0..180} retention[day]
# Net LTV: apply tracking_pct only (your case)
# CAC: CPI (your case)
# =============================
def ltv_excel_weights(r1, r7, r14, r30, r180, arpdau, tracking_pct, cpi):
    lifespan = r1 * 2.5 + r7 * 7.0 + r14 * 12.0 + r30 * 57.5 + r180 * 100.0
    ltv_gross = lifespan * arpdau
    ltv_net = ltv_gross * (1.0 - tracking_pct)
    cac = cpi
    return lifespan, ltv_gross, ltv_net, cac


def ltv_curve_sum(ret_curve, arpdau, tracking_pct, cpi):
    # default to 180-day lifespan (if horizon < 180, use last available)
    H = min(180, len(ret_curve) - 1)
    lifespan = float(np.sum(ret_curve[: H + 1]))  # include day0
    ltv_gross = lifespan * arpdau
    ltv_net = ltv_gross * (1.0 - tracking_pct)
    cac = cpi
    return lifespan, ltv_gross, ltv_net, cac


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="ROI Tool (P&L / LTV / %DAU)", page_icon="📊", layout="wide")
st.title("📊 ROI Tool (P&L / LTV / %DAU)")

with st.sidebar:
    st.header("⚙️ UA & Monetization ($)")

    horizon_days = st.number_input("Horizon (days)", min_value=30, max_value=365, value=180, step=10)

    run_full = st.checkbox("UA chạy liên tục tới hết Horizon", value=True)
    if run_full:
        ua_days = int(horizon_days)
        st.caption(f"UA Days = {ua_days}")
    else:
        ua_days = st.number_input("UA days", min_value=1, max_value=int(horizon_days), value=min(30, int(horizon_days)))

    cost_per_day = st.number_input("Cost per day ($/day)", min_value=0.0, value=1000.0, step=100.0)
    cpi = st.number_input("CPI ($/install)", min_value=0.01, value=4.0, step=0.05)
    arpdau = st.number_input("ARPDAU ($/DAU/day)", min_value=0.0, value=0.40, step=0.01)

    tracking_pct = st.number_input("Tracking cost (% of Revenue)", min_value=0.0, max_value=50.0, value=2.5, step=0.1) / 100.0

    st.divider()
    st.markdown("### 📉 Retention Inputs (%)")

    with st.expander("Chỉnh Retention (D1 → D120)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            d1 = st.number_input("D1 (%)", 0.0, 100.0, 45.0, 0.5)
            d7 = st.number_input("D7 (%)", 0.0, 100.0, 22.0, 0.5)
            d30 = st.number_input("D30 (%)", 0.0, 100.0, 12.0, 0.5)
            d60 = st.number_input("D60 (%)", 0.0, 100.0, 9.0, 0.5)
            d120 = st.number_input("D120 (%)", 0.0, 100.0, 7.0, 0.5)
        with c2:
            d3 = st.number_input("D3 (%)", 0.0, 100.0, 30.0, 0.5)
            d14 = st.number_input("D14 (%)", 0.0, 100.0, 18.0, 0.5)
            d45 = st.number_input("D45 (%)", 0.0, 100.0, 10.0, 0.5)
            d90 = st.number_input("D90 (%)", 0.0, 100.0, 8.0, 0.5)

    enforce_mono = st.checkbox("Ép Retention giảm dần (khuyên dùng)", value=True)

# Core compute
ret_curve = build_retention_curve(
    d1, d3, d7, d14, d30, d45, d60, d90, d120,
    horizon_days=int(horizon_days),
    enforce_monotonic=enforce_mono,
)

df = simulate_ua(
    cost_per_day=cost_per_day,
    cpi=cpi,
    arpdau=arpdau,
    ua_days=int(ua_days),
    horizon_days=int(horizon_days),
    retention_curve=ret_curve,
    tracking_pct=float(tracking_pct),
)

payback_day = find_payback_day(df)

total_niu = df["NIU"].sum()
total_cost = float(df["Cum_Cost"].iloc[-1])
total_rev_net = float(df["Cum_Revenue"].iloc[-1])
final_roas = (total_rev_net / total_cost) if total_cost > 0 else float("nan")
final_pnl = float(df["Cum_PnL"].iloc[-1])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total NIU", f"{total_niu:,.0f}")
k2.metric("Total Cost (UA)", f"${total_cost:,.0f}")
k3.metric("Total Revenue (Net)", f"${total_rev_net:,.0f}")
k4.metric("ROAS (cum, Net)", f"{final_roas:.2f}x" if total_cost > 0 else "N/A")
k5.metric("Cum P&L (Net)", f"${final_pnl:,.0f}")

if payback_day > 0:
    st.success(f"✅ Payback Day: Day {payback_day}")
else:
    st.warning("⚠️ Chưa hồi vốn trong horizon hiện tại.")

tab_pl, tab_ltv, tab_dau = st.tabs(["P&L", "LTV", "%DAU (30 days)"])


# =============================
# TAB: P&L
# =============================
with tab_pl:
    st.subheader("💹 P&L (Daily & Cumulative)")

    # Chart 1: DAU vs Revenue/day
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(x=df["Day"], y=df["DAU"], mode="lines", name="DAU"))
    fig_daily.add_trace(
        go.Scatter(
            x=df["Day"],
            y=df["Revenue_Net"],
            mode="lines",
            name="Revenue / day (Net)",
            yaxis="y2",
            line=dict(dash="dot"),
            opacity=0.75,
        )
    )
    fig_daily.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Day",
        yaxis=dict(title="DAU", rangemode="tozero"),
        yaxis2=dict(title="Revenue/day (Net, $)", overlaying="y", side="right", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    # Chart 2: Cumulative Cost vs Revenue vs PnL
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=df["Day"], y=df["Cum_Cost"], mode="lines", name="Cumulative Cost (UA)"))
    fig_cum.add_trace(go.Scatter(x=df["Day"], y=df["Cum_Revenue"], mode="lines", name="Cumulative Revenue (Net)", line=dict(dash="dot")))
    fig_cum.add_trace(go.Scatter(x=df["Day"], y=df["Cum_PnL"], mode="lines", name="Cumulative P&L (Net)"))

    if payback_day > 0:
        y_pb = float(df.loc[df["Day"] == payback_day, "Cum_Revenue"].iloc[0])
        fig_cum.add_trace(
            go.Scatter(
                x=[payback_day],
                y=[y_pb],
                mode="markers+text",
                name="Payback",
                text=[f"Day {payback_day}"],
                textposition="bottom right",
            )
        )

    fig_cum.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Day",
        yaxis_title="Amount ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Chart 3: Daily PnL bar
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Bar(x=df["Day"], y=df["Daily_PnL"], name="Daily P&L (Net)"))
    fig_pnl.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Day",
        yaxis_title="Daily P&L ($)",
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    with st.expander("📉 Retention curve preview", expanded=False):
        days = np.arange(0, int(horizon_days) + 1)
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(x=days, y=ret_curve * 100.0, mode="lines", name="Retention (fitted)"))
        fig_ret.add_trace(
            go.Scatter(
                x=[1, 3, 7, 14, 30, 45, 60, 90, 120],
                y=[d1, d3, d7, d14, d30, d45, d60, d90, d120],
                mode="markers+text",
                name="Anchors",
                text=[f"D{x}" for x in [1, 3, 7, 14, 30, 45, 60, 90, 120]],
                textposition="top center",
            )
        )
        fig_ret.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Day", yaxis_title="Retention (%)")
        st.plotly_chart(fig_ret, use_container_width=True)

    st.markdown("#### 📋 Table")
    st.dataframe(
        df[["Day", "Cost_UA", "NIU", "DAU", "Revenue_Gross", "TrackingCost", "Revenue_Net", "Daily_PnL", "Cum_PnL", "ROAS_cum"]],
        use_container_width=True,
        hide_index=True,
    )


# =============================
# TAB: LTV
# =============================
with tab_ltv:
    st.subheader("🧮 LTV (NIU-based)")

    method = st.radio(
        "Chọn cách tính Lifespan",
        ["Curve sum (recommended)", "Excel weights (match sheet)"],
        horizontal=True,
    )

    r1 = d1 / 100.0
    r7 = d7 / 100.0
    r14 = d14 / 100.0
    r30 = d30 / 100.0
    r180 = float(ret_curve[min(180, len(ret_curve) - 1)])

    if method.startswith("Curve"):
        lifespan, ltv_gross, ltv_net, cac = ltv_curve_sum(ret_curve, arpdau, tracking_pct, cpi)
    else:
        lifespan, ltv_gross, ltv_net, cac = ltv_excel_weights(r1, r7, r14, r30, r180, arpdau, tracking_pct, cpi)

    profit_per_user = ltv_net - cpi

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Lifespan", f"{lifespan:.2f}")
    m2.metric("LTV (Gross)", f"${ltv_gross:.2f}")
    m3.metric("LTV (Net)", f"${ltv_net:.2f}")
    m4.metric("CPI", f"${cpi:.2f}")
    m5.metric("Profit / user (NetLTV - CPI)", f"${profit_per_user:.2f}")

    st.caption(f"Tracking cost applied: {tracking_pct*100:.2f}% of revenue. CAC = CPI (your case). R180 (est) = {r180*100:.2f}%")

    st.divider()
    st.markdown("### 🎯 Profit target → NIU/Revenue (theo logic sheet, đổi NRU → NIU)")

    c1, c2 = st.columns(2)
    with c1:
        profit_target = st.number_input("Profit target ($)", min_value=0.0, value=0.0, step=1000.0)
    with c2:
        months = st.number_input("Months (sheet thường dùng 24)", min_value=1, value=24, step=1)

    if profit_target > 0:
        if profit_per_user <= 0:
            st.error("Net LTV ≤ CPI → profit_per_user <= 0, không thể đạt profit target theo mô hình hiện tại.")
        else:
            total_niu_need = profit_target / profit_per_user
            niu_per_month = total_niu_need / months
            niu_per_day = niu_per_month / 30.0

            s1, s2, s3 = st.columns(3)
            s1.metric("Total NIU cần", f"{total_niu_need:,.0f}")
            s2.metric("NIU / month", f"{niu_per_month:,.0f}")
            s3.metric("NIU / day", f"{niu_per_day:,.0f}")

            st.caption("Nếu bạn còn muốn block DAU dự kiến theo %DAU comeback (như sheet), mình có thể bật thêm dưới dạng Advanced.")
    else:
        st.info("Nhập Profit target để tính NIU cần.")

    with st.expander("Advanced (tuỳ chọn): DAU dự kiến theo %DAU comeback", expanded=False):
        st.caption("Công thức sheet: DAU ≈ NIU/day / (1 - %DAU).")
        pct_dau_comeback = st.number_input("%DAU comeback (vd 0.76)", min_value=0.0, max_value=0.99, value=0.76, step=0.01)
        if profit_target > 0 and profit_per_user > 0:
            niu_per_day = (profit_target / profit_per_user) / months / 30.0
            dau_est = niu_per_day / (1.0 - pct_dau_comeback)
            rev_day = dau_est * arpdau
            rev_month = rev_day * 30.0
            ua_cost_month = (niu_per_day * 30.0) * cpi
            budget_pct = ua_cost_month / rev_month if rev_month > 0 else float("nan")

            a, b, c = st.columns(3)
            a.metric("DAU dự kiến", f"{dau_est:,.0f}")
            b.metric("Rev/day", f"${rev_day:,.2f}")
            c.metric("%Budget", f"{budget_pct*100:.1f}%" if not math.isnan(budget_pct) else "N/A")


# =============================
# TAB: %DAU (30 days) with auto DAU/NIU + override
# =============================
with tab_dau:
    st.subheader("📌 %DAU – Bảng 30 ngày (Auto DAU/NIU + Override theo ngày)")
    st.caption("Công thức: %DAU(t) = (DAU(t) - NIU(t)) / DAU(t-1)")

    dau_day0 = st.number_input(
        "DAU ngày 0 (hôm qua trước Day1)",
        min_value=0.0,
        value=50000.0,
        step=100.0,
    )

    cta1, cta2, cta3 = st.columns([0.33, 0.33, 0.34])
    with cta1:
        auto_dau = st.checkbox("Auto DAU từ simulator", value=True)
    with cta2:
        auto_niu = st.checkbox("Auto NIU = Cost/CPI", value=True)
    with cta3:
        st.caption("Override: nhập số vào ô ngày đó. Để trống = auto.")

    model_dau = df.loc[:29, "DAU"].to_numpy()
    model_niu = df.loc[:29, "NIU"].to_numpy()

    if "dau_30_input" not in st.session_state:
        st.session_state.dau_30_input = pd.DataFrame(
            {"Day": np.arange(1, 31, dtype=int), "DAU": [np.nan] * 30, "NIU": [np.nan] * 30}
        )

    b1, b2, b3 = st.columns(3)
    if b1.button("Fill empty DAU (from model)"):
        m = st.session_state.dau_30_input["DAU"].isna()
        st.session_state.dau_30_input.loc[m, "DAU"] = model_dau[m.to_numpy()]
    if b2.button("Fill empty NIU (from model)"):
        m = st.session_state.dau_30_input["NIU"].isna()
        st.session_state.dau_30_input.loc[m, "NIU"] = model_niu[m.to_numpy()]
    if b3.button("Reset ALL overrides (clear inputs)"):
        st.session_state.dau_30_input["DAU"] = np.nan
        st.session_state.dau_30_input["NIU"] = np.nan

    edited = st.data_editor(
        st.session_state.dau_30_input,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Day": st.column_config.NumberColumn("Day", disabled=True),
            "DAU": st.column_config.NumberColumn("DAU (override)", min_value=0.0, step=1.0),
            "NIU": st.column_config.NumberColumn("NIU (override)", min_value=0.0, step=1.0),
        },
        hide_index=True,
        key="dau_30_editor",
    )
    st.session_state.dau_30_input = edited

    eff_dau = edited["DAU"].to_numpy(dtype=float)
    eff_niu = edited["NIU"].to_numpy(dtype=float)

    if auto_dau:
        eff_dau = np.where(np.isfinite(eff_dau), eff_dau, model_dau)
    if auto_niu:
        eff_niu = np.where(np.isfinite(eff_niu), eff_niu, model_niu)

    returning = eff_dau - eff_niu

    dau_prev = np.empty(30, dtype=float)
    dau_prev[:] = np.nan
    dau_prev[0] = float(dau_day0)
    for i in range(1, 30):
        dau_prev[i] = eff_dau[i - 1]

    pct_dau = np.where((dau_prev > 0) & np.isfinite(dau_prev), returning / dau_prev, np.nan)

    out = pd.DataFrame(
        {
            "Day": np.arange(1, 31, dtype=int),
            "DAU_effective": eff_dau,
            "NIU_effective": eff_niu,
            "Returning": returning,
            "DAU_prev": dau_prev,
            "%DAU": pct_dau * 100.0,
        }
    )

    st.markdown("### Kết quả")
    st.dataframe(
        out.round({"DAU_effective": 0, "NIU_effective": 0, "Returning": 0, "DAU_prev": 0, "%DAU": 2}),
        use_container_width=True,
        hide_index=True,
    )

    valid = pd.Series(pct_dau).replace([np.inf, -np.inf], np.nan).dropna()
    kk1, kk2, kk3, kk4 = st.columns(4)
    kk1.metric("Avg %DAU (valid)", f"{(valid.mean()*100):.2f}%" if len(valid) else "—")
    kk2.metric("Min %DAU", f"{(valid.min()*100):.2f}%" if len(valid) else "—")
    kk3.metric("Max %DAU", f"{(valid.max()*100):.2f}%" if len(valid) else "—")
    kk4.metric("Days usable", f"{int(np.isfinite(pct_dau).sum())}/30")

    st.markdown("### Biểu đồ")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=out["Day"], y=out["DAU_effective"], mode="lines+markers", name="DAU"))
    fig.add_trace(go.Bar(x=out["Day"], y=out["NIU_effective"], name="NIU"))
    fig.add_trace(
        go.Scatter(
            x=out["Day"],
            y=out["%DAU"],
            mode="lines+markers",
            name="%DAU",
            yaxis="y2",
            line=dict(dash="dot"),
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Day",
        yaxis=dict(title="DAU / NIU", rangemode="tozero"),
        yaxis2=dict(title="%DAU", overlaying="y", side="right", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV (%DAU 30 days)", data=csv, file_name="dau_30days.csv", mime="text/csv")
