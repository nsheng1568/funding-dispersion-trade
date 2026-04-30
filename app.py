import streamlit as st

st.set_page_config(
    page_title="Funding Dispersion Trade",
    page_icon="📈",
    layout="wide",
)

st.title("Funding Dispersion Trade")
st.markdown(
    "Use the sidebar to navigate between **Strategy State** (live account) "
    "and **Analytics** (signals, betas, portfolio)."
)