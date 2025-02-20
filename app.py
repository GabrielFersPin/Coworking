import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import folium_static

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("/workspaces/Coworking/src/results/MergedPlacesScoreDistance.csv")

df = load_data()

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("/workspaces/Coworking/src/results/random_forest_model.pkl")

model = load_model()

# Sidebar Filters
st.sidebar.header("Filter Coworking Spaces")
selected_country = st.sidebar.selectbox("Select Country", df["Country"].unique())
filtered_cities = df[df["Country"] == selected_country]["City"].unique()
selected_city = st.sidebar.selectbox("Select City", filtered_cities)
filtered_neighborhoods = df[(df["Country"] == selected_country) & (df["City"] == selected_city)]["Neightboorhood"].unique()
selected_neighborhood = st.sidebar.selectbox("Select Neighborhood", filtered_neighborhoods)

max_price = st.sidebar.slider("Max Day Pass Price", float(df["Day Pass"].min()), float(df["Day Pass"].max()), float(df["Day Pass"].mean()))

# Filter Data
filtered_df = df[(df["Day Pass"] <= max_price) &
                 (df["Country"] == selected_country) &
                 (df["City"] == selected_city) &
                 (df["Neightboorhood"] == selected_neighborhood)]

# Display Data
st.title("Coworking Space Finder")
st.dataframe(filtered_df.head())

# Map Visualization
st.header(f"Coworking Spaces in {selected_neighborhood}, {selected_city}, {selected_country}")
map_center = [filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()]
coworking_map = folium.Map(location=map_center, zoom_start=12)

for _, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=row['User Rating Count'] / 10,
        popup=f"{row['name']}<br>Rating: {row['Rating']}<br>User Rating Count: {row['User Rating Count']}",
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(coworking_map)

# Render the map
folium_static(coworking_map)

# Recommendation System
st.subheader("Top Recommended Coworking Spaces")
df_filtered = df[(df["Country"] == selected_country) & (df["City"] == selected_city) & (df["Neightboorhood"] == selected_neighborhood)]
df_filtered["Score"] = (max_price - df_filtered["Day Pass"]) + (df_filtered["Rating"] * 10) - df_filtered["distance_from_center"]
recommended_spaces = df_filtered.sort_values(by="Score", ascending=False).head(5)
st.dataframe(recommended_spaces)

st.write("### Developed by Gabriel Pinheiro")
st.write("### [LinkedIn](https://www.linkedin.com/in/gabriel-pinheiro-7b4a8a1b1/)")
st.write("### [GitHub](https://github.com/GabrielFersPin?tab=repositories)")
