"""
Page 1: live account state + equity history.

Live data (positions, equity, funding rates) is fetched directly from the
Hyperliquid read-only API — no private key required.
Historical data is read from data/history/ CSVs committed by the ETL cron.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from hyperliquid.info import Info
from hyperliquid.utils import constants

from src.data import hl_client

VAULT_ADDRESS = "0x6e48fcE48934317b52b6CcACe381b4548683D156"
HISTORY_DIR   = Path("data/history")

st.set_page_config(page_title="Strategy State", layout="wide")
st.title("Strategy State")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def load_live_state() -> dict:
    info  = Info(constants.MAINNET_API_URL, skip_ws=True)
    state = info.user_state(VAULT_ADDRESS)

    universe = hl_client.get_universe()  # funding rates + mark prices

    equity       = float(state["marginSummary"]["accountValue"])
    margin_used  = float(state["marginSummary"].get("totalMarginUsed", 0))

    rows = []
    for item in state["assetPositions"]:
        p   = item["position"]
        szi = float(p["szi"])
        if szi == 0:
            continue
        coin = p["coin"]
        sign = 1 if szi > 0 else -1

        rows.append({
            "Coin":           coin,
            "Side":           "LONG" if szi > 0 else "SHORT",
            "Notional ($)":   round(sign * float(p["positionValue"]), 2),
            "Entry Px":       float(p["entryPx"]),
            "Mark Px":        float(universe.loc[coin, "markPx"]) if coin in universe.index else None,
            "Unr. PnL ($)":   round(float(p.get("unrealizedPnl", 0)), 2),
            "8h Funding (%)": round(float(universe.loc[coin, "funding"]) * 100, 4)
                              if coin in universe.index else None,
        })

    positions = pd.DataFrame(rows)
    gross   = sum(abs(r["Notional ($)"]) for r in rows)
    net     = sum(r["Notional ($)"] for r in rows)
    unr_pnl = sum(r["Unr. PnL ($)"] for r in rows)

    return dict(equity=equity, margin_used=margin_used, gross=gross,
                net=net, unr_pnl=unr_pnl, positions=positions)


@st.cache_data(ttl=600)
def load_equity_history() -> pd.DataFrame:
    path = HISTORY_DIR / "equity.csv"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "equity"])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=600)
def load_position_history() -> pd.DataFrame:
    path = HISTORY_DIR / "positions.csv"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "coin", "notional"])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

state = load_live_state()

# --- Metrics row ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Equity",         f"${state['equity']:,.2f}")
c2.metric("Gross Exposure", f"${state['gross']:,.2f}")
c3.metric("Net Exposure",   f"${state['net']:,.2f}")
c4.metric("Margin Used",    f"${state['margin_used']:,.2f}")
c5.metric("Unrealized PnL", f"${state['unr_pnl']:+,.2f}",
          delta_color="normal" if state["unr_pnl"] >= 0 else "inverse")

st.divider()

# --- Open positions ---
st.subheader("Open Positions")
if state["positions"].empty:
    st.info("No open positions.")
else:
    pos = state["positions"].copy()
    pos["Notional ($)"] = pos["Notional ($)"].map("${:,.2f}".format)
    pos["Unr. PnL ($)"] = pos["Unr. PnL ($)"].map("${:+,.2f}".format)
    st.dataframe(pos, use_container_width=True, hide_index=True)

st.divider()

# --- Equity history chart ---
st.subheader("Equity History")
eq_hist = load_equity_history()
if eq_hist.empty:
    st.info("No equity history yet — run the ETL cron at least once.")
else:
    fig = px.area(
        eq_hist, x="timestamp", y="equity",
        labels={"equity": "Equity (USDC)", "timestamp": ""},
    )
    fig.update_traces(line_color="#00b4d8", fillcolor="rgba(0,180,216,0.15)")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Position history ---
st.subheader("Position History")
pos_hist = load_position_history()
if pos_hist.empty:
    st.info("No position history yet.")
else:
    # Show most recent snapshot per coin as a small table, plus a timeline chart
    # of notional over time for each coin that has been held.
    coins_held = pos_hist["coin"].dropna().unique()
    if len(coins_held) > 0:
        pivot = (
            pos_hist.dropna(subset=["coin"])
            .pivot_table(index="timestamp", columns="coin", values="notional", aggfunc="last")
            .fillna(0)
            .reset_index()
        )
        fig2 = px.line(
            pivot.melt(id_vars="timestamp", var_name="coin", value_name="notional"),
            x="timestamp", y="notional", color="coin",
            labels={"notional": "Signed Notional ($)", "timestamp": ""},
        )
        fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
        st.plotly_chart(fig2, use_container_width=True)

st.caption(f"Live data cached 2 min · history cached 10 min · "
           f"vault `{VAULT_ADDRESS[:10]}...`")