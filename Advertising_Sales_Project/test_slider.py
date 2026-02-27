import streamlit as st

st.sidebar.slider('plain int', 0, 100000, 15000, step=5000)
st.sidebar.slider('format str %d', 0, 100000, 15000, step=5000, format="%d")
st.sidebar.slider('format localized', 0, 100000, 15000, step=5000, format="localized")
st.sidebar.slider('format commas', 0, 100000, 15000, step=5000, format="%d")
