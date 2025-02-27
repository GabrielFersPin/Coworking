import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import folium
import os

## Load the trained model
model_path = "/workspaces/Coworking/src/results/random_forest_model.pkl"
day_pass_model = joblib.load(model_path)
month_pass_model = joblib.load('/workspaces/Coworking/src/results/month_ridge_params_model.pkl')
df_final = pd.read_csv("/workspaces/Coworking/src/results/PreprocessedData.csv")
df = pd.read_csv('/workspaces/Coworking/src/results/MergedPlacesScoreDistance.csv')
# Extract feature names used during training
day_trained_features = day_pass_model.feature_names_in_
month_trained_features = month_pass_model.feature_names_in_

# Load encoders
city_encoder = joblib.load("/workspaces/Coworking/src/results/city_encoder.pkl")

# Streamlit app
st.title("Coworking Space Recomendation System")

# User selects a country
selected_country = st.selectbox("Select a country:", df["Country"].unique())

# User selects a city
selected_city = st.selectbox("Select a city:", df[df["Country"] == selected_country]["City"].unique())

# Function to predict day pass price
def predict_pass_price(city, city_encoder, model, day_trained_features):
    # Create an empty DataFrame with training features, initializing all values as zero
    input_data = pd.DataFrame(columns=day_trained_features)
    input_data.loc[0] = 0  # Initialize all values to zero

    # One-hot encode the City
    city_encoded = city_encoder.transform(pd.DataFrame({"City": [city]}))
    city_encoded_df = pd.DataFrame(city_encoded, columns=city_encoder.get_feature_names_out())

    # Ensure city encoding is applied to the correct columns
    for col in city_encoded_df.columns:
        if col in input_data.columns:
            input_data[col] = city_encoded_df[col].values[0]

    # Set placeholder values for required numerical features
    default_values = {
        "log_population": 13.8,  
        "log_income": 11.5,
        "log_distance": 1.0,
        "income_per_capita": 50000
    }
    
    for feature in default_values:
        if feature in input_data.columns:
            input_data[feature] = default_values[feature]

    # Convert to float (RandomForest requires float32/float64)
    input_data = input_data.astype(float)

    # Debugging
    print("Input Data:", input_data)

    # Make a prediction
    prediction = model.predict(input_data)
    
    # Debugging
    print("Prediction:", prediction)

    return prediction[0]

#Function to predict month pass price
def predict_month_pass_price(city, city_encoder, month_pass_model, month_trained_features):
    # Create an empty DataFrame with training features, initializing all values as zero
    input_data = pd.DataFrame(columns=month_trained_features)
    input_data.loc[0] = 0  # Initialize all values to zero

    # One-hot encode the City
    city_encoded = city_encoder.transform(pd.DataFrame({"City": [city]}))
    city_encoded_df = pd.DataFrame(city_encoded, columns=city_encoder.get_feature_names_out())

    # Ensure city encoding is applied to the correct columns
    for col in city_encoded_df.columns:
        if col in input_data.columns:
            input_data[col] = city_encoded_df[col].values[0]

    # Set placeholder values for required numerical features
    default_values = {
        "log_population": 13.8,  
        "log_income": 11.5,
        "log_distance": 1.0,
        "income_per_capita": 50000
    }
    
    for feature in default_values:
        if feature in input_data.columns:
            input_data[feature] = default_values[feature]

    # Convert to float (Model requires float32/float64)
    input_data = input_data.astype(float)

    # Debugging
    print("Input Data:", input_data)

    # Make a prediction
    prediction = month_pass_model.predict(input_data)
    
    # Debugging
    print("Prediction:", prediction)

    return prediction[0]

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

# Predict day pass price
if st.button("Predict Pass Prices"):
    # Predict both Day Pass and Month Pass Prices
    predicted_day_price = predict_pass_price(selected_city, city_encoder, day_pass_model, day_trained_features)
    predicted_month_price = predict_month_pass_price(selected_city, city_encoder, month_pass_model, month_trained_features)
    
    # Display the predictions
    st.write(f"Predicted Day Pass Price: **${predicted_day_price:.2f}**")
    st.write(f"Predicted Month Pass Price: **${predicted_month_price:.2f}**")

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