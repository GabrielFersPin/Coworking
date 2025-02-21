import streamlit as st
import pandas as pd
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
st.sidebar.header("Select Your Preferences")

# Dropdown filters for preferences
selected_country = st.sidebar.selectbox("Select Country", df["Country"].unique())
filtered_data = df[df["Country"] == selected_country]

if filtered_data.empty:
    st.warning("No coworking spaces found for the selected country.")
    st.stop()

selected_city = st.sidebar.selectbox("Select City", filtered_data["City"].unique())
filtered_data = filtered_data[filtered_data["City"] == selected_city]

if filtered_data.empty:
    st.warning("No coworking spaces found for the selected city.")
    st.stop()

# User preferences
st.sidebar.header("Your Preferences")
min_rating = st.sidebar.slider("Minimum Rating", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
min_rating_count = st.sidebar.slider("Minimum Rating Count", min_value=0, max_value=500, value=50)
max_distance = st.sidebar.slider("Maximum Distance from City Center (km)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
min_transport = st.sidebar.slider("Minimum Transport Accessibility", min_value=0, max_value=10, value=5)

# Filter data based on user preferences
recommended_spaces = filtered_data[
    (filtered_data["Rating"] >= min_rating) &
    (filtered_data["User Rating Count"] >= min_rating_count) &
    (filtered_data["distance_from_center"] <= max_distance) &
    (filtered_data["Transport"] >= min_transport)
]

# Sort by score (or any other metric you prefer)
recommended_spaces = recommended_spaces.sort_values(by="Score", ascending=False)

# Display Recommendations
st.title("Recommended Coworking Spaces in " + selected_city)

if recommended_spaces.empty:
    st.warning("No coworking spaces match your preferences.")
else:
    st.dataframe(recommended_spaces[["name", "Neighborhood", "Rating", "User Rating Count", "distance_from_center", "Transport", "Score"]].head(20))

    # Map Visualization
    st.header(f"Recommended Coworking Spaces in {selected_city}")

    # Calculate map center
    map_center = [recommended_spaces["Latitude"].mean(), recommended_spaces["Longitude"].mean()]
    coworking_map = folium.Map(location=map_center, zoom_start=12)

    # Add recommended coworking spaces to the map
    for _, row in recommended_spaces.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['name']}<br>Rating: {row['Rating']}<br>"
                  f"User Rating Count: {row['User Rating Count']}<br>"
                  f"Distance from Center: {row['distance_from_center']} km<br>"
                  f"Transport: {row['Transport']}",
            icon=folium.Icon(color="blue"),
        ).add_to(coworking_map)

    # Render the map
    folium_static(coworking_map)

st.write("### Developed by Gabriel Pinheiro")
st.write("### [LinkedIn](https://www.linkedin.com/in/gabriel-pinheiro-7b4a8a1b1/)")
st.write("### [GitHub](https://github.com/GabrielFersPin?tab=repositories)")