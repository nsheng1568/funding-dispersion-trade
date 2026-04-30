import streamlit as st

st.set_page_config(layout="wide")

pg = st.navigation([
    st.Page("pages/1_Strategy_State.py", title="Strategy State"),
    st.Page("pages/2_Analytics.py",      title="Analytics"),
])
pg.run()
