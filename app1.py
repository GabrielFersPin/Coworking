import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import folium
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import tempfile
from streamlit_folium import folium_static
from dotenv import load_dotenv

load_dotenv()

# Set the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the data using relative paths
df = pd.read_csv(os.path.join(base_dir, "src", "results", "MergedPlacesScoreDistance.csv"))
# Load the amenities-based coworking data
amenities_df = pd.read_json(os.path.join(base_dir, "/workspaces/Coworking/src/data_processing/merged_coworking_spaces.json"), lines=True)

# Load models
day_pass_model = joblib.load(os.path.join(base_dir, "models", "day_ridge_model.pkl"))
month_pass_model = joblib.load(os.path.join(base_dir, "models", "month_ridge_model.pkl"))

# Streamlit app
st.title("Coworking Space Recommendation System")

# User selects a country
selected_country = st.selectbox("Select a country:", df["Country"].unique())

# User selects a city
selected_city = st.selectbox("Select a city:", df[df["Country"] == selected_country]["City"].unique())

# Filtering Section
st.sidebar.header("Filter Coworking Spaces")
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 4.0, 0.1)
min_rating_count = st.sidebar.number_input("Min. User Rating Count", min_value=0, value=10, step=5)
max_distance = st.sidebar.slider("Max Distance from Center (km)", 0.0, 20.0, 5.0, 0.1)
min_transport = st.sidebar.slider("Min Transport Score", 0, 10, 5, 1)

# Filter coworking spaces
def filter_data(df, country, city, min_rating, min_rating_count, max_distance, min_transport):
    filtered_data = df[(df["Country"] == country) & (df["City"] == city)]
    if filtered_data.empty:
        return pd.DataFrame()
    
    recommended_spaces = filtered_data[
        (filtered_data["Rating"] >= min_rating) &
        (filtered_data["User Rating Count"] >= min_rating_count) &
        (filtered_data["distance_from_center"] <= max_distance) &
        (filtered_data["Transport"] >= min_transport)
    ]
    return recommended_spaces.sort_values(by="Score", ascending=False)

city = selected_city
country = selected_country

# Get recommendations
recommended_spaces = filter_data(df, country, city, min_rating, min_rating_count, max_distance, min_transport)

# Display recommendations
st.title(f"Recommended Coworking Spaces in {city}")


if recommended_spaces.empty:
    st.warning("No coworking spaces match your preferences.")
else:
    st.dataframe(recommended_spaces[["name", "Neighborhood", "Rating", "User Rating Count", "distance_from_center", "Transport", 'Day Pass', 'Month Pass']].head(20))

# Display coworking spaces on a map
st.header(f"Map of Recommended Spaces in {city}")

if not recommended_spaces.empty:
    # Create the map
    map_center = [recommended_spaces["Latitude"].mean(), recommended_spaces["Longitude"].mean()]
    coworking_map = folium.Map(location=map_center, zoom_start=12)

    for _, row in recommended_spaces.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['name']}<br>Rating: {row['Rating']}",
            icon=folium.Icon(color="blue"),
        ).add_to(coworking_map)
    
    folium_static(coworking_map)
    

# Display reference prices for comparison
city_data = df[df['City'] == selected_city]
if not city_data.empty:
    avg_day = city_data['Day Pass'].mean()
    avg_month = city_data['Month Pass'].mean()
    st.markdown(
    f"**For reference:**\n\nAverage prices in **{selected_city}** are:\n"
    f"- **Day pass:** ${avg_day:.2f}\n"
    f"- **Month pass:** ${avg_month:.2f}"
    )
