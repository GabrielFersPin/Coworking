import streamlit as st
import pandas as pd
import re

# Load the data
df = pd.read_csv("/workspaces/Coworking/src/data_processing/merged_coworking_spaces.csv") 

# Clean price column
df["price_numeric"] = df["price"].apply(
    lambda x: float(re.search(r"\d+(?:[\.,]\d+)?", str(x)).group().replace(",", ".")) 
    if pd.notnull(x) and re.search(r"\d+", str(x)) else None
)

# First, inspect which rows have prices that are just digits
invalid_price_rows = df[df["price"].str.strip().isin(["1", "2", "3", "4", "5"])]

# Then, filter them out
df = df[~df["price"].str.strip().isin(["1", "2", "3", "4", "5"])]


# Load amenities dataframe (amenities_df contains only amenities data)
amenities_df = pd.read_csv('/workspaces/Coworking/src/data_processing/extracted_amenities.csv')

selected_city = st.sidebar.selectbox("Choose a city", sorted(df["city"].dropna().unique()))

# List of available cities in the 'df' (which contains city data)
cities = df["city"].unique().tolist()

# Filter the dataframe by selected city (from 'df' DataFrame)
city_df = df[df["city"] == selected_city]

# Extract amenity columns from the amenities dataframe (from extracted_amenities.csv)
amenity_columns = [col for col in amenities_df.columns if col.startswith("has_amenity_")]
amenity_labels = [col.replace("has_amenity_", "").replace("_", " ").title() for col in amenity_columns]
amenity_map = dict(zip(amenity_labels, amenity_columns))

st.sidebar.header("Filter by Amenities")

# Get most common amenities dynamically from the amenities_df
common_amenities = amenities_df[amenity_columns].sum().sort_values(ascending=False)

# Let user select city for amenities filtering (but it's just an option since the city is in the 'df')
# The amenities_df may contain data for multiple cities, so this is useful for filtering
selected_city_amenity = st.sidebar.selectbox("Select a city (for amenities)", cities)

# Add a 'city' column to amenities_df by merging with city_df (assuming amenities_df and df have the same length)
amenities_df["city"] = df["city"]

# Filter amenities dataframe by selected city
filtered_amenities_df = amenities_df[amenities_df["city"] == selected_city_amenity]

# Let user choose preferred amenities
selected_amenities = st.sidebar.multiselect(
    "Select preferred amenities:",
    common_amenities.index.tolist()
)

# Merge amenities_df (which now has amenities + city) with df (which has name, description, etc.)
full_amenities_df = pd.concat([df[["name", "description"]], amenities_df], axis=1)

filtered_amenities_df = full_amenities_df[full_amenities_df["city"] == selected_city_amenity]

for amenity in selected_amenities:
    filtered_amenities_df = filtered_amenities_df[filtered_amenities_df[amenity] == 1]

# Create a new column 'price_numeric' by extracting the number
df["price_numeric"] = df["price"].apply(lambda x: float(re.search(r"\d+(?:[\.,]\d+)?", str(x)).group().replace(",", ".")) if pd.notnull(x) and re.search(r"\d+", str(x)) else None)

# 💸 Price slider (This part is for filtering based on price in the city dataframe)
max_price = int(city_df["price_numeric"].max())
user_price = st.sidebar.slider("Maximum price (€ / month)", 0, max_price, max_price)

# --- Filtering Logic ---
# Start with the city filtered dataframe
filtered_df = city_df.copy()

# Filter by amenities in the city_df
for amenity in selected_amenities:
    filtered_df = filtered_df[filtered_df[amenity_map[amenity]] == 1]

# Filter by price (ensure it's numeric)
filtered_df = filtered_df[filtered_df["price_numeric"] <= user_price]

# --- Show Results ---
st.title("Coworking Recommendation System")
st.subheader(f"Results for: {selected_city}")

if filtered_df.empty:
    st.warning("No coworking spaces found with the selected options 😕")
else:
    for _, row in filtered_df.iterrows():
        st.markdown(f"### {row['name']}")
        st.markdown(f"**Price:** {row['price']}")
        st.markdown(f"[🌐 Visit Website]({row['url']})")
        st.markdown("---")
