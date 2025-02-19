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
    df = pd.read_csv("MergedPlacesScoreDistance.csv")
    return df

df = load_data()

# Load trained model
@st.cache_resource
def load_model():
    model = joblib.load("/workspaces/Coworking/src/results/random_forest_model.pkl")  # Replace with your actual model file
    return model

model = load_model()

# Sidebar Filters
st.sidebar.header("Filter Coworking Spaces")
max_price = st.sidebar.slider("Max Day Pass Price", float(df["Day Pass"].min()), float(df["Day Pass"].max()), float(df["Day Pass"].mean()))
selected_city = st.sidebar.selectbox("Select City", df["Country"].unique())

# Filter Data
filtered_df = df[(df["Day Pass"] <= max_price) & (df["Country"] == selected_city)]

# Display Data
st.title("Coworking Space Finder")
st.dataframe(filtered_df)

# Map Visualization
## Load data
def load_data():
    return pd.read_csv("/workspaces/Coworking/src/results/AllPlaces.csv")

data = load_data()

# Sidebar filters
st.sidebar.header("Filters")
selected_country = st.sidebar.selectbox("Select Country", data["Country"].unique())
filtered_data = data[data["Country"] == selected_country]

# Map display
st.header(f"Coworking Spaces in {selected_country}")
map_center = [filtered_data["Latitude"].mean(), filtered_data["Longitude"].mean()]
coworking_map = folium.Map(location=map_center, zoom_start=10)

for _, row in filtered_data.iterrows():
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

# Price Prediction
st.subheader("Price Prediction for New Spaces")
user_population = st.number_input("Enter Population", min_value=0, value=int(df["Population"].mean()))
user_income = st.number_input("Enter Median Household Income", min_value=0, value=int(df["Median Household Income"].mean()))
distance = st.number_input("Enter Distance from Center", min_value=0.0, value=float(df["distance_from_center"].mean()))

if st.button("Predict Price"):
    features = np.array([[user_population, user_income, distance]])  # Adjust based on your model
    predicted_price = model.predict(features)[0]
    st.success(f"Predicted Day Pass Price: ${predicted_price:.2f}")

# Recommendation System
st.subheader("Top Recommended Coworking Spaces")
df["Score"] = (max_price - df["Day Pass"]) + (df["Rating"] * 10) - df["distance_from_center"]
recommended_spaces = df.sort_values(by="Score", ascending=False).head(5)
st.dataframe(recommended_spaces)

st.write("### Developed by [Your Name] - Powered by Machine Learning 🚀")
