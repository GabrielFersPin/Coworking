import streamlit as st
import pandas as pd
import ast
import re
import numpy as np
from collections import Counter
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set page configuration
st.set_page_config(layout="wide", page_title="Coworking Space Finder")

# Set paths to the data files
amenities_file_path = "/workspaces/Coworking/src/data_processing/extracted_amenities.csv"
coworking_file_path = "/workspaces/Coworking/src/data_processing/merged_coworking_spaces.csv"
top_rated_file_path = "/workspaces/Coworking/src/results/MergedPlacesScoreDistance.csv"

# Check if the files exist
if not os.path.exists(amenities_file_path):
    st.error(f"Error: The amenities file {amenities_file_path} does not exist.")
    st.stop()
if not os.path.exists(coworking_file_path):
    st.error(f"Error: The coworking spaces file {coworking_file_path} does not exist.")
    st.stop()

# Load datasets
amenities_df = pd.read_csv(amenities_file_path)
coworking_df = pd.read_csv(coworking_file_path)

# Try to load top rated spaces if file exists
has_top_rated = os.path.exists(top_rated_file_path)
if has_top_rated:
    top_rated_df = pd.read_csv(top_rated_file_path)
else:
    st.warning(f"Top rated spaces file not found at {top_rated_file_path}")

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
most_common_amenities = amenity_counts.most_common(15)

# Create amenity columns for filtering and clustering
for amenity in set(all_amenities):
    df[f'has_amenity_{amenity}'] = df['amenities_list'].apply(
        lambda x: 1 if isinstance(x, list) and amenity in x else 0
    )

# Function to detect sequential price patterns like "1 2 3 4" 
def is_sequential_price(price):
    if pd.isna(price):
        return False
    
    # Convert the price to string for pattern matching
    price_str = str(price).strip()
    
    # Check for sequential numbers with spaces between them
    # This pattern matches things like "1 2 3", "10 11 12", etc.
    seq_pattern = re.compile(r'\b\d+\s+\d+\s+\d+\b')
    if seq_pattern.search(price_str):
        return True
    
    # Also check for any price that has more than 3 numbers separated by spaces
    # This would catch patterns like "1 4 7 10" that aren't strictly sequential
    if len(re.findall(r'\b\d+\b', price_str)) > 3:
        return True
    
    return False

# Convert price to numeric if it's not already, removing sequential price patterns
if 'price' in df.columns:
    # First mark invalid sequential prices
    df['is_invalid_price'] = df['price'].apply(is_sequential_price)
    
    # Extract numeric prices only for valid prices
    df['price_numeric'] = np.nan  # Initialize with NaN
    
    for idx, row in df.iterrows():
        if not row['is_invalid_price']:
            # Extract numeric values from valid price strings
            if not pd.isna(row['price']):
                price_match = re.search(r'(\d+(?:\.\d+)?)', str(row['price']))
                if price_match:
                    df.at[idx, 'price_numeric'] = float(price_match.group(1))

# Build recommendation system
def build_recommendation_system(df):
    # Create feature matrix from amenities and other features
    feature_columns = [col for col in df.columns if col.startswith('has_amenity_')]
    
    # Add price as a feature if available (normalized)
    if 'price_numeric' in df.columns:
        # Create a copy of price_numeric to avoid modifying the original
        price_col = df['price_numeric'].copy()
        # Fill NaN values with median to avoid issues with normalization
        price_col = price_col.fillna(price_col.median())
        # Check if there's variance in price (avoid division by zero)
        if price_col.max() > price_col.min():
            df['price_normalized'] = (price_col - price_col.min()) / (price_col.max() - price_col.min())
            feature_columns.append('price_normalized')
    
    # Calculate similarity between spaces
    features_matrix = df[feature_columns].fillna(0)
    
    # Only compute similarity if we have enough data
    if len(features_matrix) > 1:
        similarity_matrix = cosine_similarity(features_matrix)
        return similarity_matrix
    else:
        return None

# Function to get recommendations
def get_recommendations(space_idx, similarity_matrix, df, n=5):
    # Get top N similar spaces
    similar_spaces = list(enumerate(similarity_matrix[space_idx]))
    similar_spaces = sorted(similar_spaces, key=lambda x: x[1], reverse=True)
    similar_spaces = similar_spaces[1:n+1]  # Skip the first one (itself)
    
    # Return the similar spaces information with similarity scores
    recommended_spaces = []
    for i, score in similar_spaces:
        space_info = df.iloc[i].copy()
        space_info['similarity_score'] = score
        recommended_spaces.append(space_info)
    
    return recommended_spaces

# App title and description
st.title("Coworking Space Finder and Analysis")
st.write("Find and analyze coworking spaces based on price and amenities")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Find Spaces", "Similar Spaces", "Top Rated Spaces"])

with tab1:
    # Sidebar for filters
    st.sidebar.header("Filter your coworking preferences")

    # City filter if available
    if 'city' in df.columns:
        cities = sorted(df['city'].dropna().unique().tolist())
        selected_city = st.sidebar.selectbox("Select city", ["All Cities"] + cities, key="city_filter_tab1")

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
    selected_labels = st.sidebar.multiselect("Select amenities you want", sorted_amenity_labels, key="amenity_filter_tab1")

    # Price slider if price data is available
    if 'price_numeric' in df.columns and not df['price_numeric'].isna().all():
        max_price = int(df["price_numeric"].dropna().max())
        default_price = max_price
        user_price = st.sidebar.slider("Maximum price (€ / month)", 0, max_price, default_price, key="price_filter_tab1")
        has_price_filter = True
    else:
        has_price_filter = False
        st.sidebar.info("Price filtering not available for this dataset")

    # Filter dataframe based on user input
    filtered_df = df.copy()
    
    # Remove rows with invalid sequential prices
    if 'is_invalid_price' in filtered_df.columns:
        filtered_df = filtered_df[~filtered_df['is_invalid_price']]

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
                    
                if "address" in row and not pd.isna(row["address"]):
                    st.write(f"📍 {row['address']}")
                
                if "price" in row and not pd.isna(row["price"]) and not row.get('is_invalid_price', False):
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

with tab2:
    st.header("Find Similar Coworking Spaces")
    
    # First, filter out rows with invalid price data or missing names
    recommendation_df = df.copy()
    if 'is_invalid_price' in recommendation_df.columns:
        recommendation_df = recommendation_df[~recommendation_df['is_invalid_price']]
    
    if 'name' in recommendation_df.columns:
        recommendation_df = recommendation_df.dropna(subset=['name'])
    
    if len(recommendation_df) < 2:
        st.warning("Not enough data to build a recommendation system.")
    else:
        # Build the recommendation system
        similarity_matrix = build_recommendation_system(recommendation_df)
        
        if similarity_matrix is None:
            st.warning("Couldn't build the recommendation system due to limited data features.")
        else:
            # Create an interactive exploration section
            st.subheader("Select a coworking space to find similar options")
            
            # City filter for recommendations
            if 'city' in recommendation_df.columns:
                rec_cities = sorted(recommendation_df['city'].dropna().unique().tolist())
                selected_rec_city = st.selectbox("Filter by city", ["All Cities"] + rec_cities)
                
                if selected_rec_city != "All Cities":
                    city_spaces = recommendation_df[recommendation_df['city'] == selected_rec_city]
                else:
                    city_spaces = recommendation_df
            else:
                city_spaces = recommendation_df
            
            # Select a space to find similar ones
            space_options = city_spaces['name'].tolist()
            
            if space_options:
                selected_space = st.selectbox("Select a coworking space", space_options)
                
                # Get the index of the selected space
                selected_idx = city_spaces[city_spaces['name'] == selected_space].index[0]
                
                # Get the recommendations
                num_recommendations = st.slider("Number of recommendations", 1, 10, 5)
                recommendations = get_recommendations(selected_idx, similarity_matrix, recommendation_df, n=num_recommendations)
                
                # Display the selected space details
                st.subheader(f"Selected Space: {selected_space}")
                
                selected_row = recommendation_df.loc[selected_idx]
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if "city" in selected_row and not pd.isna(selected_row["city"]):
                        st.write(f"🏙️ {selected_row['city']}")
                        
                    if "address" in selected_row and not pd.isna(selected_row["address"]):
                        st.write(f"📍 {selected_row['address']}")
                    
                    if "price" in selected_row and not pd.isna(selected_row["price"]):
                        st.write(f"💸 {selected_row['price']}")
                    
                    # Display amenities for this space
                    if isinstance(selected_row['amenities_list'], list) and len(selected_row['amenities_list']) > 0:
                        st.write("✨ Amenities: " + ", ".join(amenity.replace('_', ' ').title() for amenity in selected_row['amenities_list']))
                
                with col2:
                    if "url" in selected_row and not pd.isna(selected_row["url"]):
                        st.markdown(f"[🔗 View Website]({selected_row['url']})")
                        
                # Show description if available
                if "description" in selected_row and not pd.isna(selected_row["description"]):
                    with st.expander("See description"):
                        st.write(selected_row["description"])
                
                st.markdown("---")
                
                # Display similar spaces
                st.subheader(f"Similar coworking spaces to {selected_space}")
                
                for i, space in enumerate(recommendations):
                    col1, col2, col3 = st.columns([2.5, 0.5, 1])
                    
                    with col1:
                        st.subheader(space["name"])
                        
                        if "city" in space and not pd.isna(space["city"]):
                            st.write(f"🏙️ {space['city']}")
                            
                        if "address" in space and not pd.isna(space["address"]):
                            st.write(f"📍 {space['address']}")
                        
                        if "price" in space and not pd.isna(space["price"]):
                            st.write(f"💸 {space['price']}")
                        
                        # Display amenities for this space
                        if isinstance(space['amenities_list'], list) and len(space['amenities_list']) > 0:
                            st.write("✨ Amenities: " + ", ".join(amenity.replace('_', ' ').title() for amenity in space['amenities_list']))
                    
                    with col2:
                        # Display similarity score as percentage
                        similarity = space['similarity_score'] * 100
                        st.metric("Match", f"{similarity:.1f}%")
                    
                    with col3:
                        if "url" in space and not pd.isna(space["url"]):
                            st.markdown(f"[🔗 View Website]({space['url']})")
                    
                    # Show description if available
                    if "description" in space and not pd.isna(space["description"]):
                        with st.expander("See description"):
                            st.write(space["description"])
                    
                    st.markdown("---")
                
                # Show visualization of similarities
                st.subheader("Visualization of Similar Spaces")
                
                # Create visualization of similarities
                if 'price_numeric' in recommendation_df.columns:
                    # Get all spaces to show in viz (selected + recommendations)
                    viz_spaces = [selected_idx] + [recommendation_df[recommendation_df['name'] == rec['name']].index[0] for rec in recommendations]
                    viz_df = recommendation_df.iloc[viz_spaces].copy()
                    
                    # Mark the selected space
                    viz_df['type'] = 'Similar Space'
                    viz_df.loc[selected_idx, 'type'] = 'Selected Space'
                    
                    # Calculate total amenities
                    amenity_cols = [col for col in viz_df.columns if col.startswith('has_amenity_')]
                    viz_df['amenity_count'] = viz_df[amenity_cols].sum(axis=1)
                    
                    # Create scatter plot
                    fig = px.scatter(
                        viz_df, 
                        x='price_numeric',
                        y='amenity_count',
                        color='type',
                        hover_name='name',
                        size_max=20,
                        color_discrete_map={
                            'Selected Space': '#FF4B4B',
                            'Similar Space': '#1E88E5'
                        },
                        labels={
                            'price_numeric': 'Price (€)',
                            'amenity_count': 'Number of Amenities'
                        }
                    )
                    
                    # Make selected space point larger
                    fig.update_traces(
                        marker=dict(size=15),
                        selector=dict(name='Selected Space')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create amenity comparison chart
                st.subheader("Amenity Comparison")
                
                # Get most common amenities among this set
                comparison_spaces = [selected_row] + recommendations
                comparison_amenities = []
                for space in comparison_spaces:
                    if isinstance(space['amenities_list'], list):
                        comparison_amenities.extend(space['amenities_list'])
                
                common_amenities = [amenity for amenity, count in Counter(comparison_amenities).most_common(10)]
                
                # Create comparison data
                comparison_data = []
                for i, space in enumerate([selected_row] + recommendations):
                    space_name = space['name']
                    space_type = "Selected" if i == 0 else "Similar"
                    
                    for amenity in common_amenities:
                        has_amenity = amenity in space['amenities_list'] if isinstance(space['amenities_list'], list) else False
                        comparison_data.append({
                            'space': space_name,
                            'amenity': amenity.replace('_', ' ').title(),
                            'has_amenity': 'Yes' if has_amenity else 'No',
                            'type': space_type
                        })
                
                # Convert to DataFrame
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create heatmap
                if not comparison_df.empty and len(common_amenities) > 0:
                    pivot_df = comparison_df.pivot_table(
                        index='space', 
                        columns='amenity',
                        values='has_amenity',
                        aggfunc=lambda x: 1 if 'Yes' in x.values else 0
                    )
                    
                    # Sort rows so selected space is first
                    pivot_df = pivot_df.reset_index()
                    pivot_df['is_selected'] = pivot_df['space'] == selected_space
                    pivot_df = pivot_df.sort_values('is_selected', ascending=False).drop('is_selected', axis=1)
                    pivot_df = pivot_df.set_index('space')
                    
                    # Create heatmap
                    fig, ax = plt.subplots(figsize=(10, len(pivot_df) * 0.5 + 2))
                    sns.heatmap(
                        pivot_df, 
                        cmap=['#f5f5f5', '#4CAF50'],
                        cbar=False,
                        linewidths=1,
                        linecolor='white',
                        ax=ax
                    )
                    ax.set_title('Amenity Comparison')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("No spaces available with the selected filter.")

with tab3:
    st.header("Top Rated Coworking Spaces")
    
    if not has_top_rated:
        st.warning("Top rated spaces data not available")
    else:
        # City filter for top rated tab
        if 'city' in top_rated_df.columns:
            top_rated_cities = sorted(top_rated_df['city'].dropna().unique().tolist())
            selected_top_rated_city = st.selectbox("Select city", ["All Cities"] + top_rated_cities, key="city_filter_tab3")
            
            # Filter by city if selected
            if selected_top_rated_city != "All Cities":
                filtered_top_rated = top_rated_df[top_rated_df['city'] == selected_top_rated_city]
            else:
                filtered_top_rated = top_rated_df
        else:
            filtered_top_rated = top_rated_df
        
        # First show the map for the selected city
        if 'latitude' in filtered_top_rated.columns and 'longitude' in filtered_top_rated.columns:
            st.subheader(f"Map of Top Coworking Spaces {f'in {selected_top_rated_city}' if selected_top_rated_city != 'All Cities' else ''}")
            
            # Filter out rows with missing coordinates
            map_data = filtered_top_rated.dropna(subset=['latitude', 'longitude'])
            
            if not map_data.empty:
                # Add ranking to name for better map identification
                map_data = map_data.sort_values('score', ascending=False).reset_index(drop=True)
                map_data['display_name'] = map_data.apply(lambda x: f"#{x.name + 1} {x['name']}", axis=1)
                
                # Create a dataframe with coordinates for mapping
                map_df = pd.DataFrame({
                    'lat': map_data['latitude'],
                    'lon': map_data['longitude'],
                    'name': map_data['display_name']
                })
                
                # Display map with increased height for better visibility
                st.map(map_df, use_container_width=True, zoom=12)
            else:
                st.info("No location data available for mapping")
        
        # Display as city groups
        if 'city' in filtered_top_rated.columns:
            # Group by city
            cities = filtered_top_rated['city'].unique()
            
            for city in cities:
                st.subheader(f"Top Coworking Spaces in {city}")
                city_spaces = filtered_top_rated[filtered_top_rated['city'] == city].sort_values('score', ascending=False).head(5)
                
                for i, (_, row) in enumerate(city_spaces.iterrows()):
                    # Create a card with ranking
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        # Display rank with trophy for #1
                        if i == 0:
                            st.markdown(f"<h1 style='text-align: center; color: gold;'>🏆</h1>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h2 style='text-align: center;'>#{i+1}</h2>", unsafe_allow_html=True)
                        
                        # Display score
                        if 'score' in row:
                            score = row['score']
                            st.markdown(f"<h3 style='text-align: center;'>{score:.1f}⭐</h3>", unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader(row["name"] if "name" in row else "Coworking Space")
                        
                        if "address" in row and not pd.isna(row["address"]):
                            st.write(f"📍 {row['address']}")
                        
                        if "price" in row and not pd.isna(row["price"]):
                            st.write(f"💸 {row['price']}")
                        
                        # Show distance if available
                        if "distance" in row and not pd.isna(row["distance"]):
                            st.write(f"🚶 {row['distance']:.2f}km from city center")
                        
                        # Add website link
                        if "url" in row and not pd.isna(row["url"]):
                            st.markdown(f"[🔗 View Website]({row['url']})")
                        
                        # Show all available amenities if they exist
                        amenities_col = next((col for col in row.index if col.endswith('amenities_list')), None)
                        if amenities_col and isinstance(row[amenities_col], list) and len(row[amenities_col]) > 0:
                            st.write("✨ **Amenities:**")
                            amenities_text = ", ".join(amenity.replace('_', ' ').title() for amenity in row[amenities_col])
                            st.write(amenities_text)
                        
                        # Add expandable section for more details
                        with st.expander("More Details"):
                            # Show any additional information that might be available
                            for col_name, value in row.items():
                                # Skip columns we've already displayed or that are internal
                                if col_name in ['name', 'address', 'price', 'distance', 'url', 'latitude', 'longitude', 
                                                'score', 'city', 'amenities_list', 'index'] or col_name.startswith('has_amenity_'):
                                    continue
                                
                                # Skip empty values
                                if pd.isna(value) or value == '' or value == []:
                                    continue
                                
                                # Format column name for display
                                display_name = col_name.replace('_', ' ').title()
                                
                                # Handle different data types appropriately
                                if isinstance(value, (int, float)):
                                    st.write(f"**{display_name}:** {value}")
                                elif isinstance(value, list):
                                    st.write(f"**{display_name}:** {', '.join(str(item) for item in value)}")
                                else:
                                    st.write(f"**{display_name}:** {value}")
                            
                            # Add any ratings information if available
                            ratings_col = next((col for col in row.index if 'rating' in col.lower()), None)
                            if ratings_col and not pd.isna(row[ratings_col]):
                                st.write(f"**User Rating:** {row[ratings_col]}⭐")
                            
                            # Add any reviews count if available
                            reviews_col = next((col for col in row.index if 'review' in col.lower()), None)
                            if reviews_col and not pd.isna(row[reviews_col]):
                                st.write(f"**Number of Reviews:** {row[reviews_col]}")
                    
                    st.markdown("---")
        else:
            # Simple list without city grouping
            st.subheader("Top Rated Coworking Spaces")
            
            top_spaces = filtered_top_rated.sort_values('Score', ascending=False).head(5)
            
            for i, (_, row) in enumerate(top_spaces.iterrows()):
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    # Display rank with trophy for #1
                    if i == 0:
                        st.markdown(f"<h1 style='text-align: center; color: gold;'>🏆</h1>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='text-align: center;'>#{i+1}</h2>", unsafe_allow_html=True)
                    
                    # Display score
                    if 'score' in row:
                        score = row['score'] 
                        st.markdown(f"<h3 style='text-align: center;'>{score:.1f}⭐</h3>", unsafe_allow_html=True)
                
                with col2:
                    st.subheader(row["name"] if "name" in row else "Coworking Space")
                    
                    if "address" in row and not pd.isna(row["address"]):
                        st.write(f"📍 {row['address']}")
                    
                    if "price" in row and not pd.isna(row["price"]):
                        st.write(f"💸 {row['price']}")
                    
                    # Add website link
                    if "url" in row and not pd.isna(row["url"]):
                        st.markdown(f"[🔗 View Website]({row['url']})")
                    
                    # Show all available amenities if they exist
                    amenities_col = next((col for col in row.index if col.endswith('amenities_list')), None)
                    if amenities_col and isinstance(row[amenities_col], list) and len(row[amenities_col]) > 0:
                        st.write("✨ **Amenities:**")
                        amenities_text = ", ".join(amenity.replace('_', ' ').title() for amenity in row[amenities_col])
                        st.write(amenities_text)
                    
                    # Add expandable section for more details
                    with st.expander("More Details"):
                        # Show any additional information that might be available
                        for col_name, value in row.items():
                            # Skip columns we've already displayed or that are internal
                            if col_name in ['name', 'address', 'price', 'url', 'latitude', 'longitude', 
                                            'score', 'amenities_list', 'index'] or col_name.startswith('has_amenity_'):
                                continue
                            
                            # Skip empty values
                            if pd.isna(value) or value == '' or value == []:
                                continue
                            
                            # Format column name for display
                            display_name = col_name.replace('_', ' ').title()
                            
                            # Handle different data types appropriately
                            if isinstance(value, (int, float)):
                                st.write(f"**{display_name}:** {value}")
                            elif isinstance(value, list):
                                st.write(f"**{display_name}:** {', '.join(str(item) for item in value)}")
                            else:
                                st.write(f"**{display_name}:** {value}")
                
                st.markdown("---")