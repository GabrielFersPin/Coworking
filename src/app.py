import streamlit as st

# Set page configuration
st.set_page_config(page_title="Coworking Space Insights", layout="wide")

# App title
st.title("Coworking Space Market Analysis")

# Introduction
st.markdown("""
Welcome to my interactive dashboard app! Here, you'll find insights into coworking space trends across Spain.
These dashboards are designed to help identify key opportunities and provide actionable data for decision-making.
""")

# Embed Power BI dashboards
st.header("Dashboard 1: Customer Sentiment Analysis")
st.markdown("""
<iframe width="900" height="600" src="YOUR_POWER_BI_EMBED_LINK_1" frameborder="0" allowFullScreen="true"></iframe>
""", unsafe_allow_html=True)

st.header("Dashboard 2: Location-Based Insights")
st.markdown("""
<iframe width="900" height="600" src="YOUR_POWER_BI_EMBED_LINK_2" frameborder="0" allowFullScreen="true"></iframe>
""", unsafe_allow_html=True)

# Add a footer or call-to-action
st.markdown("""
---
**Connect with me on [LinkedIn](www.linkedin.com/in/gabriel-fernandes-pinheiro-728628248)** or leave a comment below with your thoughts!
""")

