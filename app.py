import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("/workspaces/Coworking/src/results/MergedPlacesScoreDistance.csv")

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Coworking Spaces")

# Dropdown filters
selected_country = st.sidebar.selectbox("Select Country", df["Country"].unique())

# Filter data by country first
filtered_data = df[df["Country"] == selected_country]

if filtered_data.empty:
    st.warning("No coworking spaces found for the selected country.")
    st.stop()

selected_city = st.sidebar.selectbox("Select City", filtered_data["City"].unique())

# Filter further by city
filtered_data = filtered_data[filtered_data["City"] == selected_city]

if filtered_data.empty:
    st.warning("No coworking spaces found for the selected city.")
    st.stop()

# Display Data
st.title("Top 5 Coworking Spaces in " + selected_city)
st.dataframe(filtered_data[["name", "Neighborhood", "Rating", "User Rating Count"]])

# Display Ratings
st.header("Ratings Overview")
st.write("Average Rating: ", filtered_data["Rating"].mean())
st.write("Average User Rating Count: ", filtered_data["User Rating Count"].mean())

# Display prices
st.header("Prices Overview")
st.write("Day Pass Price: ", filtered_data["Day Pass"].mean())
st.write("Monthly Price: ", filtered_data["Month Pass"].mean())

# Display Prices Distribution
st.header("Prices Distribution")
fig = px.histogram(filtered_data, y="Day Pass", nbins=5, title="Day Pass Price Distribution")
st.plotly_chart(fig)

fig = px.histogram(filtered_data, y="Month Pass", nbins=5, title="Monthly Price Distribution")
st.plotly_chart(fig)

# Display Transport connectivity
st.header("Transport Connectivity Overview")
st.write("Number of conections with public transport: ", filtered_data[['name', "Transport"]])

# Display Distance to the city center
st.header("Distance to the city center Overview")
st.write("Distance to the city center: ", filtered_data[['name', "distance_from_center"]])

# Display the score of the place
st.header("Score Overview")
st.write("Score: ", filtered_data["Score"])

# Map Visualization
st.header(f"Coworking Spaces in {selected_city}")

# Calculate map center
map_center = [filtered_data["Latitude"].mean(), filtered_data["Longitude"].mean()]
coworking_map = folium.Map(location=map_center, zoom_start=12)

# Add coworking spaces to the map
for _, row in filtered_data.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['name']}<br>Rating: {row['Rating']}<br>"
              f"User Rating Count: {row['User Rating Count']}",
        icon=folium.Icon(color="blue"),
    ).add_to(coworking_map)

# Render the map
folium_static(coworking_map)

st.write("### Developed by Gabriel Pinheiro")
st.write("### [LinkedIn](https://www.linkedin.com/in/gabriel-pinheiro-7b4a8a1b1/)")
st.write("### [GitHub](https://github.com/GabrielFersPin?tab=repositories)")
