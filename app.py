import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import folium
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load the data
df = pd.read_csv("/workspaces/Coworking/src/results/MergedPlacesScoreDistance.csv")
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
    map_center = [recommended_spaces["Latitude"].mean(), recommended_spaces["Longitude"].mean()]
    coworking_map = folium.Map(location=map_center, zoom_start=12)

    for _, row in recommended_spaces.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['name']}<br>Rating: {row['Rating']}<br>"
                  f"User Rating Count: {row['User Rating Count']}<br>"
                  f"Distance from Center: {row['distance_from_center']} km<br>"
                  f"Transport: {row['Transport']}",
            icon=folium.Icon(color="blue"),
        ).add_to(coworking_map)

    # Save the map as an HTML file
    map_html = "/tmp/coworking_map.html"
    coworking_map.save(map_html)

    # Display the map in Streamlit
    st.components.v1.html(open(map_html, 'r').read(), height=600)
    
# Predict the price of day pass and month pass
st.header("Predict Prices")
st.write("Please enter the details of the coworking space to predict the prices.")
st.write("If you don't know the value of a feature, leave it as 0.")

# Load the trained models
day_pass_model = joblib.load("/workspaces/Coworking/src/results/day_ridge_model.pkl")
month_pass_model = joblib.load("/workspaces/Coworking/src/results/month_ridge_model.pkl")

# Load the encoders and scaler saved during training
city_encoder = joblib.load("/workspaces/Coworking/src/results/city_encoder.pkl")
neighborhood_encoder = joblib.load("/workspaces/Coworking/src/results/neighborhood_encoder.pkl")
scaler = joblib.load("/workspaces/Coworking/src/results/minmax_scaler.pkl")

# Initialize user inputs with normal scale values
user_inputs = {}

user_inputs["City"] = selected_city
neighborhood_options = df[df["City"] == selected_city]["Neighborhood"].unique()
user_inputs["Neighborhood"] = st.selectbox("Neighborhood", neighborhood_options, key="neighborhood")
user_inputs["Rating"] = st.number_input("Rating", min_value=0.0, max_value=5.0, value=0.0, step=0.1, key="rating_price")
user_inputs["User Rating Count"] = st.number_input("User Rating Count", min_value=0, value=0, step=1, key="user_rating_count")
user_inputs["Distance from Center (km)"] = st.number_input("Distance from Center (km)", min_value=0.0, value=0.0, step=0.1, key="distance")
user_inputs["Transport"] = st.number_input("Transport Accessibility", min_value=0, value=0, step=1, key="transport")
user_inputs["income_per_capita"] = st.number_input("Income per Capita", min_value=0, value=0, step=1, key="median_income")
user_inputs["Population"] = st.number_input("Population", min_value=0, value=0, step=1, key="population")
user_inputs["Income"] = st.number_input("Median Household Income", min_value=0, value=0, step=1, key="income")

# Convert user inputs into DataFrame
input_data = pd.DataFrame([user_inputs])

# Create log-transformed features using log1p (to handle zero values)
input_data["log_income"] = np.log1p(input_data["Income"])
input_data["log_population"] = np.log1p(input_data["Population"])
# If your model was trained on log transformed distance, you could do the same:
input_data["log_distance"] = np.log1p(input_data["Distance from Center (km)"])

# Drop the original non-log columns, if they're not used by the model
input_data.drop(["Income", "Population", "Distance from Center (km)"], axis=1, inplace=True)

# One-hot encode categorical features (using the unchanged columns)
city_encoded = city_encoder.transform(input_data[['City']])
neighborhood_encoded = neighborhood_encoder.transform(input_data[['Neighborhood']])

city_encoded_df = pd.DataFrame(city_encoded, columns=city_encoder.categories_[0])
neighborhood_encoded_df = pd.DataFrame(neighborhood_encoded, columns=neighborhood_encoder.categories_[0])

# Combine the encoded columns with the transformed inputs
input_data_encoded = pd.concat(
    [input_data.drop(['City', 'Neighborhood'], axis=1), city_encoded_df, neighborhood_encoded_df],
    axis=1
)

# Apply scaling only to the log-transformed numerical columns
numerical_columns = ['log_population', 'log_income', 'log_distance', 'income_per_capita']
input_data_encoded[numerical_columns] = scaler.transform(input_data_encoded[numerical_columns])

# Filter model features to those present in input_data_encoded
day_model_features = [feat for feat in day_pass_model.feature_names_in_ if feat in input_data_encoded.columns]
month_model_features = [feat for feat in month_pass_model.feature_names_in_ if feat in input_data_encoded.columns]

input_data_day = input_data_encoded[day_model_features]
input_data_month = input_data_encoded[month_model_features]

# Reindex input_data_encoded to include all features expected by the model
input_data_day = input_data_encoded.reindex(columns=day_pass_model.feature_names_in_, fill_value=0)
input_data_month = input_data_encoded.reindex(columns=month_pass_model.feature_names_in_, fill_value=0)

# Optionally, drop unwanted columns if they exist
if 'Day Pass' in input_data_day.columns:
    input_data_day = input_data_day.drop(columns=['Day Pass'])
if 'Month Pass' in input_data_month.columns:
    input_data_month = input_data_month.drop(columns=['Month Pass'])

# Predict the prices
day_pass_price = day_pass_model.predict(input_data_day)[0]
month_pass_price = month_pass_model.predict(input_data_month)[0]

# Show the results
st.write(f"Predicted Day Pass Price: ${day_pass_price:.2f}")
st.write(f"Predicted Month Pass Price: ${month_pass_price:.2f}")