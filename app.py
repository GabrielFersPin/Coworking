import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
import joblib

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("/workspaces/Coworking/src/results/MergedPlacesScoreDistance.csv")

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

def display_recommendations(recommended_spaces, city):
    st.title(f"Recommended Coworking Spaces in {city}")
    if recommended_spaces.empty:
        st.warning("No coworking spaces match your preferences.")
    else:
        st.dataframe(recommended_spaces[["name", "Neighborhood", "Rating", "User Rating Count", "distance_from_center", "Transport", "Score"]].head(20))
        display_map(recommended_spaces, city)

def display_map(recommended_spaces, city):
    st.header(f"Recommended Coworking Spaces in {city}")
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
    folium_static(coworking_map)

def main():
    df = load_data()

    # Sidebar Filters
    st.sidebar.header("Select Your Preferences")
    selected_country = st.sidebar.selectbox("Select Country", df["Country"].unique())
    selected_city = st.sidebar.selectbox("Select City", df[df["Country"] == selected_country]["City"].unique())
    min_rating = st.sidebar.slider("Minimum Rating", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
    min_rating_count = st.sidebar.slider("Minimum Rating Count", min_value=0, max_value=500, value=50)
    max_distance = st.sidebar.slider("Maximum Distance from City Center (km)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    min_transport = st.sidebar.slider("Minimum Transport Accessibility", min_value=0, max_value=10, value=5)

    recommended_spaces = filter_data(df, selected_country, selected_city, min_rating, min_rating_count, max_distance, min_transport)
    display_recommendations(recommended_spaces, selected_city)

    # User Inputs for Price Prediction
    st.sidebar.header("Coworking Space Price Prediction")
    selected_neighborhood = st.sidebar.selectbox("Select Neighborhood", df[df["City"] == selected_city]["Neighborhood"].unique())
    transport_access = st.sidebar.slider("Transport Accessibility (1-10)", min_value=1, max_value=10, value=5)

    # Use the selected input to predict price
    predicted_daily_price = predict_price(selected_city, selected_neighborhood, transport_access)
    predicted_monthly_price = predicted_daily_price * 30  # Assuming 30 days in a month

    # Displaying the predicted daily and monthly price
    st.write(f"Predicted Daily Price: {predicted_daily_price:.2f} EUR")
    st.write(f"Predicted Monthly Price: {predicted_monthly_price:.2f} EUR")

    fig = px.histogram(df, x="Price_Daily", color="City", title="Price Distribution by City")
    st.plotly_chart(fig)    

# Load trained models
day_pass_model = joblib.load("/workspaces/Coworking/src/results/random_forest_model.pkl")
month_pass_model = joblib.load("/workspaces/Coworking/src/results/ridge_params_model.pkl")

# Load encoders
city_encoder = joblib.load("/workspaces/Coworking/src/results/city_encoder.pkl")
neighborhood_encoder = joblib.load("/workspaces/Coworking/src/results/neighborhood_encoder.pkl")

# Define the prediction function
def predict_pass_price(city, city_encoder, neighborhood_encoder, model):
    # Create an empty DataFrame with the same structure as training data
    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0  # Initialize with zeros

    # One-hot encode the city
    df_city = pd.DataFrame({'City': [city]})
    city_encoded = city_encoder.transform(df_city)
    city_encoded_df = pd.DataFrame(city_encoded, columns=city_encoder.get_feature_names_out())

    # Add encoded city columns to the input data
    for col in city_encoded_df.columns:
        if col in input_data.columns:
            input_data[col] = city_encoded_df[col].values[0]

    # Set numerical features (use real values if available)
    input_data['log_population'] = 1000000.0  # Replace with real data
    input_data['log_income'] = 50000.0  # Replace with real data
    input_data['log_distance'] = 10.0  # Replace with real data
    input_data['income_per_capita'] = 5000.0  # Replace with real data

    # Ensure correct column order
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Predict the pass price
    predicted_price = model.predict(input_data)

    return predicted_price[0]

# Streamlit UI
st.title("Coworking Pass Price Predictor")

# Select city
cities = ["Madrid", "Barcelona", "New York", "Tokyo"]
selected_city = st.selectbox("Select a city:", cities)

if st.button("Predict Prices"):
    day_pass_price = predict_pass_price(selected_city, city_encoder, neighborhood_encoder, day_pass_model)
    month_pass_price = predict_pass_price(selected_city, city_encoder, neighborhood_encoder, month_pass_model)

    st.subheader(f"Predicted Prices for {selected_city}:")
    st.write(f"**Day Pass Price:** ${day_pass_price:.2f}")
    st.write(f"**Monthly Pass Price:** ${month_pass_price:.2f}")


st.write("### Developed by Gabriel Pinheiro")
st.write("### [LinkedIn](https://www.linkedin.com/in/gabriel-pinheiro-7b4a8a1b1/)")
st.write("### [GitHub](https://github.com/GabrielFersPin?tab=repositories)")

if __name__ == "__main__":
    main()