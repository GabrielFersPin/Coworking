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


# Set the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the data using relative paths
df = pd.read_csv(os.path.join(base_dir, "src", "results", "MergedPlacesScoreDistance.csv"))

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
    
# Predict the price of day pass and month pass
st.header("Simulation for predict prices")
st.write("Please enter the details of the coworking space to predict the prices.")
st.write("If you don't know the value of a feature, leave it as 0.")

# Load the encoders and scaler saved during training
city_encoder = joblib.load(os.path.join(base_dir, "models", "city_encoder.pkl"))
neighborhood_encoder = joblib.load(os.path.join(base_dir, "models", "neighborhood_encoder.pkl"))
scaler = joblib.load(os.path.join(base_dir, "models", "minmax_scaler.pkl"))

# Initialize user inputs with normal scale values
user_inputs = {}

user_inputs["City"] = selected_city
neighborhood_options = df[df["City"] == selected_city]["Neighborhood"].unique()
user_inputs["Neighborhood"] = st.selectbox("Neighborhood", neighborhood_options, key="neighborhood")
user_inputs["Rating"] = st.number_input("Rating", min_value=0.0, max_value=5.0, value=0.0, step=0.1, key="rating_price")
user_inputs["User Rating Count"] = st.number_input("User Rating Count", min_value=0, value=0, step=1, key="user_rating_count")
user_inputs["Distance from Center (km)"] = st.number_input("Distance from Center (km)", min_value=0.0, value=0.0, step=0.1, key="distance")
user_inputs["Transport"] = st.number_input("Transport Accessibility", min_value=0, value=0, step=1, key="transport")
user_inputs["income_per_capita"] = st.number_input("Income per Capita in USD($)", min_value=0, value=0, step=1, key="median_income")
user_inputs["Population"] = st.number_input("Population", min_value=0, value=0, step=1, key="population")
user_inputs["Income"] = st.number_input("Median Household Income in USD($)", min_value=0, value=0, step=1, key="income")

def predict_prices(selected_city, selected_neighborhood, user_inputs, df, day_model, month_model, city_encoder, neighborhood_encoder, scaler):
    # Función para aplicar log sin transformar dos veces
    def safe_log_transform(value):
        return np.log1p(value) if value > 0 else 0  # Evita log(0)

    # Obtener valores de usuario o valores por defecto de la ciudad
    pop_value = user_inputs.get("Population", df[df['City'] == selected_city]['Population'].mean())
    income_value = user_inputs.get("Income", df[df['City'] == selected_city]['Median Household Income'].mean())
    distance_value = user_inputs.get("Distance from Center (km)", df[df['City'] == selected_city]['distance_from_center'].mean())

    # Aplicar transformación logarítmica
    prediction_input = {
        'City': selected_city,
        'Neighborhood': selected_neighborhood,
        'log_population': safe_log_transform(pop_value),
        'log_income': safe_log_transform(income_value),
        'log_distance': safe_log_transform(distance_value),
        'income_per_capita': income_value / pop_value if pop_value > 0 else 0,
        'Rating': user_inputs.get("Rating", 0),
        'User Rating Count': user_inputs.get("User Rating Count", 0),
        'Transport': user_inputs.get("Transport", 0)
    }

    # Convertir a DataFrame
    input_data = pd.DataFrame([prediction_input])

    # One-hot encoding
    city_encoded_df = pd.DataFrame(city_encoder.transform(input_data[['City']]), columns=city_encoder.categories_[0])
    neighborhood_encoded_df = pd.DataFrame(neighborhood_encoder.transform(input_data[['Neighborhood']]), columns=neighborhood_encoder.categories_[0])

    # Unir datos categóricos y numéricos
    input_data_encoded = pd.concat([input_data.drop(['City', 'Neighborhood'], axis=1), city_encoded_df, neighborhood_encoded_df], axis=1)

    # Escalar con MinMaxScaler (excluding specific features)
    numerical_columns = ['log_population', 'log_income', 'log_distance', 'income_per_capita']
    input_data_encoded[numerical_columns] = scaler.transform(input_data_encoded[numerical_columns])

    # Ajustar dimensiones para los modelos
    input_data_day = input_data_encoded.reindex(columns=day_model.feature_names_in_, fill_value=0)
    input_data_month = input_data_encoded.reindex(columns=month_model.feature_names_in_, fill_value=0)

    # Añadir las características no escaladas
    input_data_day['Rating'] = input_data['Rating']
    input_data_day['Transport'] = input_data['Transport']
    input_data_month['Rating'] = input_data['Rating']
    input_data_month['Transport'] = input_data['Transport']

    # Predicciones
    day_pass_price = max(day_model.predict(input_data_day)[0], 0)
    month_pass_price = max(month_model.predict(input_data_month)[0], 0)

    return day_pass_price, month_pass_price

# Predecir precios
day_pass_price, month_pass_price = predict_prices(
    selected_city, 
    user_inputs["Neighborhood"],
    user_inputs, 
    df, 
    day_pass_model, 
    month_pass_model,
    city_encoder,  
    neighborhood_encoder,
    scaler
)

# Mostrar resultados
st.write(f"Predicted Day Pass Price: ${day_pass_price:.2f}")
st.write(f"Predicted Monthly Membership: ${month_pass_price:.2f}")

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