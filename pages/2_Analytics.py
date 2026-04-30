"""
Page 2: signal rankings, beta summary, and signal history.

All data is read from pre-computed artifacts committed by the ETL cron:
  - data/coin_betas.parquet  — beta, idio_vol, r_squared per coin
  - data/history/signals.csv — timestamped signal snapshots
  - data/history/target.csv  — timestamped target portfolio snapshots
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.models.signal import MAX_LEVERAGE

HISTORY_DIR = Path("data/history")
DATA_DIR    = Path("data")

st.title("Analytics")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_betas() -> pd.DataFrame:
    path = DATA_DIR / "coin_betas.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path).astype(float)


@st.cache_data(ttl=600)
def load_signal_history() -> pd.DataFrame:
    path = HISTORY_DIR / "signals.csv"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "coin", "signal"])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=600)
def load_target_history() -> pd.DataFrame:
    path = HISTORY_DIR / "target.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=["timestamp", "long_coin", "short_coin",
                     "long_usd", "short_usd", "spread"])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


betas      = load_betas()
sig_hist   = load_signal_history()
tgt_hist   = load_target_history()

# Latest signal snapshot
latest_ts       = sig_hist["timestamp"].max() if not sig_hist.empty else None
latest_signals  = (
    sig_hist[sig_hist["timestamp"] == latest_ts].copy()
    if latest_ts is not None else pd.DataFrame(columns=["coin", "signal"])
)

tab1, tab2, tab3 = st.tabs(["Current Signal", "Beta Summary", "Signal History"])


# ---------------------------------------------------------------------------
# Tab 1 — Current Signal
# ---------------------------------------------------------------------------

with tab1:
    if latest_signals.empty:
        st.info("No signal data yet — run the ETL cron at least once.")
    else:
        latest_signals = latest_signals.sort_values("signal").reset_index(drop=True)
        n = len(latest_signals)
        long_coin  = latest_signals.iloc[0]["coin"]
        short_coin = latest_signals.iloc[-1]["coin"]
        spread     = latest_signals.iloc[-1]["signal"] - latest_signals.iloc[0]["signal"]

        # Target portfolio card
        col1, col2, col3 = st.columns(3)
        col1.metric("Long",  long_coin,  help="Lowest signal — buy funding")
        col2.metric("Short", short_coin, help="Highest signal — sell funding")
        col3.metric("Signal Spread", f"{spread:.3f}")

        if not betas.empty and long_coin in betas.index and short_coin in betas.index:
            from src.models.signal import size_position
            L, S = size_position(long_coin, short_coin, betas)
            col1.caption(f"Target leverage: {L:.2f}×")
            col2.caption(f"Target leverage: {S:.2f}×")

        st.divider()

        # Signal bar chart
        colors = ["#ef4444" if c == short_coin
                  else "#22c55e" if c == long_coin
                  else "#94a3b8"
                  for c in latest_signals["coin"]]

        fig = px.bar(
            latest_signals,
            x="signal", y="coin", orientation="h",
            color="signal",
            color_continuous_scale=[[0, "#22c55e"], [0.5, "#94a3b8"], [1, "#ef4444"]],
            labels={"signal": "Risk-Adjusted Signal", "coin": ""},
        )
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=20, b=0),
            height=max(300, n * 22),
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.3)
        st.plotly_chart(fig, width='stretch')

        st.caption(f"Snapshot as of {latest_ts:%Y-%m-%d %H:%M UTC}  ·  "
                   f"{n} coins  ·  spread = {spread:.4f}")


# ---------------------------------------------------------------------------
# Tab 2 — Beta Summary
# ---------------------------------------------------------------------------

with tab2:
    if betas.empty:
        st.info("No betas file found. Commit `data/coin_betas.parquet` to the repo "
                "after running `python -m src.models.betas`.")
    else:
        import numpy as np
        ANN = np.sqrt(365 * 3)  # 8h periods per year

        df = betas.copy().sort_values("beta", ascending=False).reset_index()
        df = df.rename(columns={"index": "Coin"})

        # Annualise per-period vols (raw cols are per 8h)
        if "idio_vol" in df.columns and "idio_vol_ann" not in df.columns:
            df["idio_vol_ann"] = df["idio_vol"] * ANN
        if "total_vol" in df.columns:
            df["total_vol_ann"] = df["total_vol"] * ANN

        # Build display table
        disp = pd.DataFrame()
        disp["Coin"]          = df["Coin"]
        disp["Beta"]          = df["beta"].round(2)
        if "idio_vol_ann" in df.columns:
            disp["Idio Vol (Ann)"]  = (df["idio_vol_ann"] * 100).round(1).astype(str) + "%"
        if "total_vol_ann" in df.columns:
            disp["Total Vol (Ann)"] = (df["total_vol_ann"] * 100).round(1).astype(str) + "%"
        if "r_squared" in df.columns:
            disp["R²"]              = df["r_squared"].round(2)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.dataframe(disp, width='stretch', hide_index=True,
                         height=min(600, len(disp) * 36 + 40))
        with col_b:
            if "idio_vol_ann" in df.columns:
                scatter_df = df.copy()
                scatter_df["idio_vol_ann_pct"] = scatter_df["idio_vol_ann"] * 100
                fig = px.scatter(
                    scatter_df,
                    x="beta", y="idio_vol_ann_pct",
                    text="Coin",
                    labels={"beta": "Market Beta",
                            "idio_vol_ann_pct": "Idio Vol (Ann)",
                            "Coin": "Coin"},
                    title="Beta vs Idiosyncratic Vol",
                )
                fig.update_traces(textposition="top center", textfont_size=9)
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=400)
                st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Tab 3 — Signal History
# ---------------------------------------------------------------------------

with tab3:
    if sig_hist.empty:
        st.info("No signal history yet.")
    else:
        # Identify coins that were ever in the top or bottom 3
        top_bottom = set()
        for ts, grp in sig_hist.groupby("timestamp"):
            grp_s = grp.sort_values("signal")
            top_bottom.update(grp_s["coin"].iloc[:3])   # bottom 3 (long candidates)
            top_bottom.update(grp_s["coin"].iloc[-3:])  # top 3 (short candidates)

        focus = sig_hist[sig_hist["coin"].isin(top_bottom)]

        fig = px.line(
            focus,
            x="timestamp", y="signal", color="coin",
            labels={"signal": "Risk-Adjusted Signal", "timestamp": "", "coin": "Coin"},
            title="Signal History — top/bottom 3 coins",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=400)
        st.plotly_chart(fig, width='stretch')

        # Target portfolio history
        if not tgt_hist.empty:
            st.subheader("Target Portfolio History")
            st.dataframe(
                tgt_hist.sort_values("timestamp", ascending=False).head(20),
                width='stretch', hide_index=True,
            )