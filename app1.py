import streamlit as st
import pandas as pd
import ast
import re
import numpy as np
from collections import Counter
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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

# Convert price to numeric if it's not already
if 'price' in df.columns:
    df['price_numeric'] = pd.to_numeric(
        df['price'].str.extract(r'(\d+(?:\.\d+)?)', expand=False), 
        errors='coerce'
    )

# App title and description
st.title("Coworking Space Finder and Analysis")
st.write("Find and analyze coworking spaces based on price and amenities")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Find Spaces", "Cluster Analysis", "Top Rated Spaces"])

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

with tab2:
    st.header("Coworking Space Clustering Analysis")
    
    # Prepare data for clustering
    # First, filter out rows with missing price data
    cluster_df = df.dropna(subset=['price_numeric']).copy()
    
    if len(cluster_df) < 5:
        st.warning("Not enough data with price information for clustering analysis.")
    else:
        # Select the most common amenities for clustering to reduce dimensionality
        top_amenities = [amenity for amenity, _ in most_common_amenities]
        cluster_features = [f'has_amenity_{amenity}' for amenity in top_amenities]
        
        # Add price to features
        cluster_features.append('price_numeric')
        
        # Create clustering dataframe with just the features we need
        X = cluster_df[cluster_features].copy()
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters using the elbow method
        with st.expander("Elbow Method for Optimal Clusters"):
            col1, col2 = st.columns([1, 3])
            with col1:
                max_clusters = min(10, len(cluster_df) // 5)  # Limit max clusters based on data size
                k_range = range(2, max_clusters + 1)
                inertias = []
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    inertias.append(kmeans.inertia_)
                
                # Default number of clusters
                default_clusters = 4
                num_clusters = st.slider("Select number of clusters", 2, max_clusters, default_clusters)
            
            with col2:
                fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
                ax.plot(k_range, inertias, 'bo-')
                ax.set_xlabel('Number of clusters')
                ax.set_ylabel('Inertia')
                ax.set_title('Elbow Method for Optimal k')
                ax.axvline(x=num_clusters, color='r', linestyle='--')
                st.pyplot(fig)
                plt.close(fig)
        
        # Perform K-means clustering with the selected number of clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        cluster_df['pca1'] = X_pca[:, 0]
        cluster_df['pca2'] = X_pca[:, 1]
        
        # Display clustering results
        st.subheader(f"Clustering Results ({num_clusters} clusters)")
        
        # Cluster visualization
        fig, ax = plt.figure(figsize=(10, 8)), plt.subplot(111)
        scatter = sns.scatterplot(
            x='pca1', 
            y='pca2', 
            hue='cluster', 
            palette='viridis', 
            data=cluster_df, 
            s=100, 
            alpha=0.7,
            ax=ax
        )
        ax.set_title('Coworking Spaces Clustering')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        
        # Add cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        ax.scatter(
            centers_pca[:, 0], 
            centers_pca[:, 1], 
            s=200, 
            marker='X', 
            c='red', 
            alpha=0.8, 
            label='Centroids'
        )
        ax.legend()
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Cluster analysis
        st.subheader("Cluster Characteristics")
        
        cluster_stats = []
        for i in range(num_clusters):
            cluster_data = cluster_df[cluster_df['cluster'] == i]
            avg_price = cluster_data['price_numeric'].mean()
            
            # Calculate amenity frequency in this cluster
            amenities_freq = {}
            for amenity in top_amenities:
                col = f'has_amenity_{amenity}'
                if col in cluster_data.columns:
                    amenities_freq[amenity] = cluster_data[col].mean() * 100  # Convert to percentage
            
            # Sort amenities by frequency
            sorted_amenities = sorted(amenities_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Get the top 5 most common amenities in this cluster
            top_cluster_amenities = sorted_amenities[:5]
            
            cluster_stats.append({
                'cluster': i,
                'size': len(cluster_data),
                'avg_price': avg_price,
                'top_amenities': top_cluster_amenities
            })
        
        # Display cluster stats
        for i, stats in enumerate(cluster_stats):
            with st.expander(f"Cluster {i} ({stats['size']} spaces)"):
                st.write(f"**Average Price:** €{stats['avg_price']:.2f} / month")
                st.write("**Top Amenities:**")
                for amenity, freq in stats['top_amenities']:
                    st.write(f"- {amenity.replace('_', ' ').title()}: {freq:.1f}%")
                
                # Sample spaces in this cluster
                st.write("**Sample Coworking Spaces in this Cluster:**")
                sample_spaces = cluster_df[cluster_df['cluster'] == i].head(3)
                for _, space in sample_spaces.iterrows():
                    st.write(f"- {space.get('name', 'Unnamed Space')} ({space.get('city', 'Unknown City')}): {space.get('price', 'Price not available')}")
        
        # Add option to explore a specific cluster
        selected_cluster = st.selectbox("Explore a specific cluster", 
                                       range(num_clusters), 
                                       format_func=lambda x: f"Cluster {x} ({cluster_stats[x]['size']} spaces)")
        
        if selected_cluster is not None:
            st.subheader(f"Coworking Spaces in Cluster {selected_cluster}")
            cluster_spaces = cluster_df[cluster_df['cluster'] == selected_cluster].sort_values('price_numeric')
            
            # Display spaces in this cluster
            for _, row in cluster_spaces.iterrows():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(row["name"] if "name" in row else "Coworking Space")
                    
                    if "city" in row and not pd.isna(row["city"]):
                        st.write(f"🏙️ {row['city']}")
                        
                    if "address" in row and not pd.isna(row["address"]):
                        st.write(f"📍 {row['address']}")
                    
                    if "price" in row and not pd.isna(row["price"]):
                        st.write(f"💸 {row['price']}")
                    
                    # Display amenities for this space
                    if isinstance(row['amenities_list'], list) and len(row['amenities_list']) > 0:
                        st.write("✨ Amenities: " + ", ".join(amenity.replace('_', ' ').title() for amenity in row['amenities_list']))
                
                with col2:
                    if "url" in row and not pd.isna(row["url"]):
                        st.markdown(f"[🔗 View Website]({row['url']})")
                
                st.markdown("---")

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
        
        # Display as city groups
        if 'city' in filtered_top_rated.columns:
            # Group by city
            cities = filtered_top_rated['city'].unique()
            
            for city in cities:
                st.subheader(f"Top Coworking Spaces in {city}")
                city_spaces = filtered_top_rated[filtered_top_rated['city'] == city].sort_values('score', ascending=False).head(5)
                
                for i, (_, row) in enumerate(city_spaces.iterrows()):
                    # Create a card with ranking
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
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
                    
                    with col3:
                        if "url" in row and not pd.isna(row["url"]):
                            st.markdown(f"[🔗 View Website]({row['url']})")
                    
                    st.markdown("---")
        else:
            # Simple list without city grouping
            st.subheader("Top Rated Coworking Spaces")
            
            for i, (_, row) in enumerate(filtered_top_rated.sort_values('Score', ascending=False).head(10).iterrows()):
                col1, col2, col3 = st.columns([1, 3, 1])
                
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
                
                with col3:
                    if "url" in row and not pd.isna(row["url"]):
                        st.markdown(f"[🔗 View Website]({row['url']})")
                
                st.markdown("---")
        
        # Add a map if coordinates are available
        if 'latitude' in filtered_top_rated.columns and 'longitude' in filtered_top_rated.columns:
            st.subheader("Locations of Top Rated Spaces")
            
            # Filter out rows with missing coordinates
            map_data = filtered_top_rated.dropna(subset=['latitude', 'longitude'])
            
            if not map_data.empty:
                # Create a dataframe with coordinates
                map_df = pd.DataFrame({
                    'lat': map_data['latitude'],
                    'lon': map_data['longitude'],
                    'name': map_data['name']
                })
                
                # Display map
                st.map(map_df)
            else:
                st.info("No location data available for mapping")