import streamlit as st
import pandas as pd
import ast
import re
from collections import Counter
import os

# Set paths to the data files
amenities_file_path = "/workspaces/Coworking/src/data_processing/extracted_amenities.csv"
coworking_file_path = "/workspaces/Coworking/src/data_processing/merged_coworking_spaces.csv"

# Check if the files exist
if not os.path.exists(amenities_file_path):
    st.error(f"Error: The amenities file {amenities_file_path} does not exist.")
    st.stop()
if not os.path.exists(coworking_file_path):
    st.error(f"Error: The coworking spaces file {coworking_file_path} does not exist.")
    st.stop()

# Load both datasets
amenities_df = pd.read_csv(amenities_file_path)
coworking_df = pd.read_csv(coworking_file_path)

# Convert string representation of lists to actual lists
def parse_amenities(amenity_str):
    if pd.isna(amenity_str) or amenity_str == '[]':
        return []
    try:
        return ast.literal_eval(amenity_str)
    except:
        # Handle strings that might be already quoted
        try:
            return ast.literal_eval(amenity_str.replace('"', ''))
        except:
            return []

# Apply the parsing function
amenities_df['amenities_list'] = amenities_df['extracted_amenities'].apply(parse_amenities)

# Merge the dataframes
# Assuming they have the same length and order, otherwise we need an ID to join on
if len(amenities_df) == len(coworking_df):
    # Direct merge if they're aligned
    df = pd.concat([coworking_df, amenities_df['amenities_list']], axis=1)
else:
    # If they aren't aligned, use reset_index to create a common column for joining
    amenities_df = amenities_df.reset_index()
    coworking_df = coworking_df.reset_index()
    df = pd.merge(coworking_df, amenities_df[['index', 'amenities_list']], on='index', how='left')

# Extract all unique amenities from the amenities lists
all_amenities = []
for amenities in df['amenities_list']:
    if isinstance(amenities, list):  # Check if it's a valid list
        all_amenities.extend(amenities)
all_unique_amenities = list(set(all_amenities))

# Function to find amenities in description text
def find_amenities_in_description(description, amenities_list):
    if pd.isna(description):
        return []
    
    description = description.lower()
    found_amenities = []
    
    # Mapping of amenity terms to standardized amenity names
    amenity_keywords = {
        'wifi': 'wifi',
        'internet': 'wifi',
        'coffee': 'coffee',
        'café': 'coffee',
        'cafe': 'coffee',
        'kitchen': 'kitchen',
        'parking': 'parking',
        'bike': 'bike_storage',
        'bicycle': 'bike_storage',
        'locker': 'locker',
        'meeting': 'meeting_rooms',
        'conference': 'meeting_rooms',
        'printer': 'printing',
        'print': 'printing',
        'rooftop': 'rooftop',
        'terrace': 'rooftop',
        '24/7': '24/7_access',
        '24h': '24/7_access',
        'lounge': 'lounge',
        'event': 'events',
        'workspace': 'dedicated_desk',
        'dedicated desk': 'dedicated_desk'
        # Add more mappings as needed
    }
    
    for keyword, amenity in amenity_keywords.items():
        if keyword in description and amenity in all_unique_amenities:
            found_amenities.append(amenity)
    
    return list(set(found_amenities))  # Remove duplicates

# Enhance amenities with descriptions
if 'description' in df.columns:
    for index, row in df.iterrows():
        desc_amenities = find_amenities_in_description(row.get('description', ''), all_unique_amenities)
        
        # Combine with existing amenities list
        current_amenities = row['amenities_list'] if isinstance(row['amenities_list'], list) else []
        combined_amenities = list(set(current_amenities + desc_amenities))
        
        # Update the amenities list
        df.at[index, 'amenities_list'] = combined_amenities

# Count amenities after enhancement
all_amenities = []
for amenities in df['amenities_list']:
    if isinstance(amenities, list):
        all_amenities.extend(amenities)

# Get the most common amenities
amenity_counts = Counter(all_amenities)
most_common_amenities = amenity_counts.most_common(10)

# Create amenity columns for filtering
for amenity in set(all_amenities):
    df[f'has_amenity_{amenity}'] = df['amenities_list'].apply(
        lambda x: 1 if isinstance(x, list) and amenity in x else 0
    )

# Convert price to numeric if it's not already
if 'price' in df.columns:
    df['price_numeric'] = pd.to_numeric(
        df['price'].str.extract(r'(\d+(?:\.\d+)?)', expand=False), 
        errors='coerce'
    )

# App title and description
st.title("Coworking Space Finder")
st.write("Find the perfect coworking space based on your preferences")

# Sidebar for filters
st.sidebar.header("Filter your coworking preferences")

# City filter if available
if 'city' in df.columns:
    cities = sorted(df['city'].dropna().unique().tolist())
    selected_city = st.sidebar.selectbox("Select city", ["All Cities"] + cities)

# Display most common amenities
st.sidebar.subheader("Most Common Amenities")
for amenity, count in most_common_amenities:
    st.sidebar.text(f"{amenity.replace('_', ' ').title()}: {count}")

# Amenity selection
amenity_labels = [amenity.replace("_", " ").title() for amenity in set(all_amenities)]
amenity_map = dict(zip(amenity_labels, set(all_amenities)))

# Sort amenities alphabetically for easier selection
sorted_amenity_labels = sorted(amenity_labels)

# Amenity multi-select
selected_labels = st.sidebar.multiselect("Select amenities you want", sorted_amenity_labels)

# Price slider if price data is available
if 'price_numeric' in df.columns and not df['price_numeric'].isna().all():
    max_price = int(df["price_numeric"].dropna().max())
    default_price = max_price
    user_price = st.sidebar.slider("Maximum price (€ / month)", 0, max_price, default_price)
    has_price_filter = True
else:
    has_price_filter = False
    st.sidebar.info("Price filtering not available for this dataset")

# Filter dataframe based on user input
filtered_df = df.copy()

# Apply city filter
if 'city' in df.columns and selected_city != "All Cities":
    filtered_df = filtered_df[filtered_df['city'] == selected_city]

# Apply amenity filters
for label in selected_labels:
    amenity = amenity_map[label]
    filtered_df = filtered_df[filtered_df[f'has_amenity_{amenity}'] == 1]

# Apply price filter if available
if has_price_filter:
    filtered_df = filtered_df[filtered_df["price_numeric"] <= user_price]

# Show results
st.header("Coworking spaces that match your preferences")
st.write(f"Found {len(filtered_df)} matching spaces")

if filtered_df.empty:
    st.warning("No coworking spaces found with the selected options 😕")
else:
    for _, row in filtered_df.iterrows():
        # Create a card-like effect for each result
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(row["name"] if "name" in row else "Coworking Space")
            
            if "city" in row and not pd.isna(row["city"]):
                st.write(f"🏙️ {row['city']}")
            
            if "price" in row and not pd.isna(row["price"]):
                st.write(f"💸 {row['price']}")
            
            # Display amenities for this space
            if isinstance(row['amenities_list'], list) and len(row['amenities_list']) > 0:
                st.write("✨ Amenities: " + ", ".join(amenity.replace('_', ' ').title() for amenity in row['amenities_list']))
        
        with col2:
            if "url" in row and not pd.isna(row["url"]):
                st.markdown(f"[🔗 View Website]({row['url']})")
        
        # Show description snippet if available
        if "description" in row and not pd.isna(row["description"]):
            with st.expander("See description"):
                st.write(row["description"])
            
        st.markdown("---")