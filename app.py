import streamlit as st
import pandas as pd

# Load your cleaned dataframe with amenities and price_numeric
df = pd.read_csv("your_cleaned_coworking_file.csv")  # or from your merged JSON

# Extract amenity columns (you can customize this list if needed)
amenity_columns = [col for col in df.columns if col.startswith("has_amenity_")]

# Convert for Streamlit-friendly names
amenity_labels = [col.replace("has_amenity_", "").replace("_", " ").title() for col in amenity_columns]
amenity_map = dict(zip(amenity_labels, amenity_columns))

# Sidebar for filters
st.sidebar.header("Filter your coworking preferences")

# Amenity multi-select
selected_labels = st.sidebar.multiselect("Select amenities you want", amenity_labels)

# Convert back to internal column names
selected_amenities = [amenity_map[label] for label in selected_labels]

# Price slider
max_price = int(df["price_numeric"].max())
user_price = st.sidebar.slider("Maximum price (€ / month)", 0, max_price, max_price)

# Filter dataframe based on user input
filtered_df = df.copy()

# Apply amenity filters
for amenity in selected_amenities:
    filtered_df = filtered_df[filtered_df[amenity] == 1]

# Apply price filter
filtered_df = filtered_df[filtered_df["price_numeric"] <= user_price]

# Show results
st.header("Coworking spaces that match your preferences")

if filtered_df.empty:
    st.warning("No coworking spaces found with the selected options 😕")
else:
    for _, row in filtered_df.iterrows():
        st.subheader(row["name"])
        st.write(f"📍 {row['address']}")
        st.write(f"💸 {row['price']}")
        st.write(f"[🔗 View Coworking Space]({row['url']})")
        st.markdown("---")
