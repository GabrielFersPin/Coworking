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
import pydeck as pdk
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib


# Set page configuration
st.set_page_config(layout="wide", page_title="Coworking Space Finder")

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to current script
amenities_file_path = os.path.join(current_dir, "src", "results", "extracted_amenities.csv")
coworking_file_path = os.path.join(current_dir, "src", "results", "merged_coworking_spaces.csv")
top_rated_file_path = os.path.join(current_dir, "src", "results", "MergedPlacesScoreDistance.csv")

# Check if the files exist
if not os.path.exists(amenities_file_path):
    st.error(f"Error: The amenities file {amenities_file_path} does not exist.")
    st.stop()
if not os.path.exists(coworking_file_path):
    st.error(f"Error: The coworking spaces file {coworking_file_path} does not exist.")
    st.stop()

# Load datasets with a loading indicator
with st.spinner("Loading datasets..."):
    amenities_df = pd.read_csv(amenities_file_path)
    coworking_df = pd.read_csv(coworking_file_path)

# Try to load top rated spaces if file exists
has_top_rated = os.path.exists(top_rated_file_path)
with st.spinner("Loading top rated spaces..."):
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
        try:
            return ast.literal_eval(amenity_str.replace('"', ''))
        except:
            return []

# Process amenities data with a loading indicator
with st.spinner("Processing amenities data..."):
    amenities_df['amenities_list'] = amenities_df['extracted_amenities'].apply(parse_amenities)
    # Remove wifi from amenities lists
    amenities_df['amenities_list'] = amenities_df['amenities_list'].apply(
        lambda x: [amenity for amenity in x if 'wifi' not in amenity.lower()]
    )

# Merge the dataframes with a loading indicator
with st.spinner("Merging datasets..."):
    if len(amenities_df) == len(coworking_df):
        df = pd.concat([coworking_df, amenities_df['amenities_list']], axis=1)
    else:
        amenities_df = amenities_df.reset_index()
        coworking_df = coworking_df.reset_index()
        df = pd.merge(coworking_df, amenities_df[['index', 'amenities_list']], on='index', how='left')

# Extract all unique amenities from the amenities lists
all_amenities = []
for amenities in df['amenities_list']:
    if isinstance(amenities, list):
        # Remove wifi from existing lists
        amenities = [amenity for amenity in amenities if 'wifi' not in amenity.lower()]
        all_amenities.extend(amenities)
all_unique_amenities = list(set(all_amenities))

# Function to find amenities in description text
def find_amenities_in_description(description, amenities_list):
    if pd.isna(description):
        return []
    description = description.lower()
    found_amenities = []
    amenity_keywords = {
        # Remove wifi-related keywords
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
    return list(set(found_amenities))

# Enhance amenities with descriptions wrapped in a spinner
with st.spinner("Enhancing amenities with descriptions..."):
    if 'description' in df.columns:
        for index, row in df.iterrows():
            desc_amenities = find_amenities_in_description(row.get('description', ''), all_unique_amenities)
            current_amenities = row['amenities_list'] if isinstance(row['amenities_list'], list) else []
            combined_amenities = list(set(current_amenities + desc_amenities))
            df.at[index, 'amenities_list'] = combined_amenities

# Count amenities after enhancement
with st.spinner("Finalizing data processing..."):
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

# Build clustering model
def build_clustering_model(df, n_clusters):
    """Build a clustering model for coworking spaces"""
    # Get amenity columns for clustering
    amenity_cols = [col for col in df.columns if col.startswith('has_amenity_')]
    
    # Create feature matrix
    features = []
    
    # Add amenities
    if amenity_cols:
        features.append(df[amenity_cols].fillna(0))
    
    # Add normalized price if available
    if 'price_numeric' in df.columns:
        price_data = df['price_numeric'].fillna(df['price_numeric'].median())
        scaler = StandardScaler()
        price_normalized = scaler.fit_transform(price_data.values.reshape(-1, 1))
        features.append(pd.DataFrame(price_normalized, columns=['price_scaled']))
    
    # Combine features
    if not features:
        return None, None
    
    feature_matrix = pd.concat(features, axis=1)
    
    # Apply clustering
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(feature_matrix)
        cluster_centers = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=feature_matrix.columns
        )
        return clusters, cluster_centers
    except:
        return None, None

def create_features_for_scoring(df):
    """
    Crea características numéricas para el modelo de scoring
    """
    features_df = df.copy()
    
    # 1. Características de amenidades (ya las tienes)
    amenity_cols = [col for col in df.columns if col.startswith('has_amenity_')]
    
    # 2. Características de precio
    if 'price_numeric' in df.columns:
        # Normalizar precio
        features_df['price_normalized'] = (df['price_numeric'] - df['price_numeric'].min()) / (df['price_numeric'].max() - df['price_numeric'].min())
        features_df['price_normalized'] = features_df['price_normalized'].fillna(features_df['price_normalized'].median())
    
    # 3. Características de ubicación (si tienes coordenadas)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Distancia al centro de la ciudad (necesitarías definir centros)
        city_centers = {
            'Madrid': (40.4168, -3.7038),
            'Barcelona': (41.3851, 2.1734),
            'Valencia': (39.4699, -0.3763),
            # Añadir más ciudades según tus datos
        }
        
        def calculate_distance_to_center(row):
            if pd.notna(row['city']) and row['city'] in city_centers:
                center_lat, center_lon = city_centers[row['city']]
                # Fórmula haversine simplificada
                lat_diff = row['latitude'] - center_lat
                lon_diff = row['longitude'] - center_lon
                return np.sqrt(lat_diff**2 + lon_diff**2)
            return np.nan
        
        features_df['distance_to_center'] = features_df.apply(calculate_distance_to_center, axis=1)
        features_df['distance_to_center'] = features_df['distance_to_center'].fillna(features_df['distance_to_center'].median())
    
    # 4. Características categóricas codificadas
    if 'city' in df.columns:
        le_city = LabelEncoder()
        features_df['city_encoded'] = le_city.fit_transform(df['city'].fillna('Unknown'))
    
    # 5. Contar total de amenidades
    features_df['total_amenities'] = features_df[amenity_cols].sum(axis=1)
    
    return features_df

# Función para entrenar modelo de scoring
# Funciones para el sistema de scoring - Añadir después de las importaciones

def create_features_for_scoring(df):
    """
    Crea características numéricas para el modelo de scoring
    """
    features_df = df.copy()
    
    # 1. Características de amenidades (ya las tienes)
    amenity_cols = [col for col in df.columns if col.startswith('has_amenity_')]
    
    # 2. Características de precio
    if 'price_numeric' in df.columns:
        # Normalizar precio
        price_data = df['price_numeric'].dropna()
        if len(price_data) > 0:
            features_df['price_normalized'] = (df['price_numeric'] - price_data.min()) / (price_data.max() - price_data.min())
            features_df['price_normalized'] = features_df['price_normalized'].fillna(0.5)
        else:
            features_df['price_normalized'] = 0.5
    else:
        features_df['price_normalized'] = 0.5
    
    # 3. Características categóricas codificadas
    if 'city' in df.columns:
        le_city = LabelEncoder()
        city_data = df['city'].fillna('Unknown')
        features_df['city_encoded'] = le_city.fit_transform(city_data)
    else:
        features_df['city_encoded'] = 0
    
    # 4. Contar total de amenidades
    if amenity_cols:
        features_df['total_amenities'] = features_df[amenity_cols].sum(axis=1)
    else:
        features_df['total_amenities'] = 0
    
    return features_df

def train_scoring_model(df):
    """
    Entrena un modelo para predecir la calidad/puntuación de un coworking
    """
    # Crear características
    features_df = create_features_for_scoring(df)
    
    # Crear un score sintético basado en múltiples factores
    # Score basado en amenidades, precio y otros factores
    features_df['synthetic_score'] = (
        features_df['total_amenities'] * 0.1 +  # Amenidades
        (1 - features_df['price_normalized']) * 0.3 +  # Precio (invertido)
        np.random.normal(2.5, 0.5, len(features_df))  # Factor base + variabilidad
    )
    
    # Normalizar a escala 1-5
    score_min = features_df['synthetic_score'].min()
    score_max = features_df['synthetic_score'].max()
    if score_max > score_min:
        features_df['synthetic_score'] = 1 + (features_df['synthetic_score'] - score_min) / (score_max - score_min) * 4
    else:
        features_df['synthetic_score'] = 3.0  # Valor por defecto
    
    # Asegurar que esté en rango 1-5
    features_df['synthetic_score'] = np.clip(features_df['synthetic_score'], 1, 5)
    
    target = 'synthetic_score'
    
    # Seleccionar características para el modelo
    feature_columns = ['total_amenities', 'price_normalized', 'city_encoded']
    
    # Añadir amenidades si existen
    amenity_cols = [col for col in features_df.columns if col.startswith('has_amenity_')]
    if amenity_cols:
        feature_columns.extend(amenity_cols[:10])  # Limitar a top 10 amenidades
    
    # Verificar que las columnas existen
    available_features = [col for col in feature_columns if col in features_df.columns]
    
    if not available_features:
        return None, None, {'error': 'No features available for training'}
    
    X = features_df[available_features].fillna(0)
    y = features_df[target]
    
    # Verificar que tenemos suficientes datos
    if len(X) < 10:
        return None, None, {'error': 'Not enough data for training'}
    
    # Dividir datos
    test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))  # Ajustar test_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, available_features, {'mse': mse, 'r2': r2}

def predict_coworking_score(model, feature_columns, new_coworking_params):
    """
    Predice el score de un nuevo coworking basado en sus parámetros
    """
    # Crear DataFrame con las características
    features = {}
    
    # Mapear parámetros de entrada a características del modelo
    for col in feature_columns:
        if col in new_coworking_params:
            features[col] = new_coworking_params[col]
        elif col.startswith('has_amenity_'):
            amenity_name = col.replace('has_amenity_', '')
            features[col] = new_coworking_params.get(f'has_{amenity_name}', 0)
        else:
            features[col] = 0
    
    features_df = pd.DataFrame([features])
    
    # Hacer predicción
    score = model.predict(features_df[feature_columns])[0]
    return max(1, min(5, score))  # Asegurar que esté en rango 1-5

# App title and description
st.title("Coworking Space Finder")
st.write("Find your perfect coworking space in just a few clicks. Filter by amenities, compare options, or discover top-rated spaces nearby.")

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Find Spaces", "Similar Spaces", "Top Rated Spaces", "Coworking Styles", "AI Scoring"])
with tab1:
    st.sidebar.header("How to use this tool")
    with st.sidebar.expander("📖 Usage Guide", expanded=True):
        st.markdown("""
        1. **Select your city** from the dropdown below
        2. **Choose key amenities** you need
        3. **Set your budget** using the price filter
        4. View matching spaces in the main panel
        
        💡 **Tip**: Start with fewer filters and add more if you get too many results
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Essential Filters")

    # City filter if available
    if 'city' in df.columns:
        cities = sorted(df['city'].dropna().unique().tolist())
        selected_city = st.sidebar.selectbox("Select city", ["All Cities"] + cities, key="city_filter_tab1")

    # Simplified amenity selection - show only top 10 most common
    top_10_amenities = [amenity.replace("_", " ").title() for amenity, _ in most_common_amenities[:10]]
    amenity_map = dict(zip(top_10_amenities, [am[0] for am in most_common_amenities[:10]]))
    selected_labels = st.sidebar.multiselect("Must-have amenities", sorted(top_10_amenities), key="amenity_filter_tab1")

    # Price filter if available
    if 'price_numeric' in df.columns and not df['price_numeric'].isna().all():
        min_price = int(df["price_numeric"].dropna().min())
        max_price = int(df["price_numeric"].dropna().max())
        
        # Simplified price brackets
        price_brackets = ["No limit", "0-200", "201-500", "501-1000", "1000+"]
        selected_bracket = st.sidebar.selectbox("Price range", price_brackets, key="price_filter_tab1")
        
        # Parse price range
        if selected_bracket == "No limit":
            price_min, price_max = 0, max_price
        elif selected_bracket == "1000+":
            price_min, price_max = 1000, max_price
        else:
            price_min = int(selected_bracket.split("-")[0])
            price_max = int(selected_bracket.split("-")[1])
        has_price_filter = True
    else:
        has_price_filter = False

    # Filter dataframe based on user input
    filtered_df = df.copy()
    
    # Remove rows with invalid sequential prices
    if 'is_invalid_price' in filtered_df.columns:
        filtered_df = filtered_df[~filtered_df['is_invalid_price']]

    # Apply city filter
    if 'city' in df.columns and selected_city != "All Cities":
        filtered_df = filtered_df[filtered_df['city'] == selected_city]
    
    # Apply price filter if available
    if has_price_filter:
        filtered_df = filtered_df[
            (filtered_df["price_numeric"] >= price_min) & 
            (filtered_df["price_numeric"] <= price_max)
        ]

    # Apply amenity filters
    for label in selected_labels:
        amenity = amenity_map[label]
        filtered_df = filtered_df[filtered_df[f'has_amenity_{amenity}'] == 1]

    # Show results
    st.header("Coworking spaces that match your preferences")
    st.subheader("Select filters on the left to find coworking spaces that match your exact needs.")
    st.write(f"Found {len(filtered_df)} matching spaces")

    if filtered_df.empty:
        st.warning("No coworking spaces found with the selected options 😕")
    else:
        for idx, row in filtered_df.iterrows():
            # Create a card-like effect for each result
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.subheader(row["name"] if "name" in row else "Coworking Space")
                if "city" in row and not pd.isna(row["city"]):
                    st.write(f"🏙️ {row['city']}")
                if "address" in row and not pd.isna(row["address"]):
                    st.write(f"📍 {row['address']}")
                if "price" in row and not pd.isna(row["price"]) and not row.get('is_invalid_price', False):
                    st.write(f"💸 {row['price']}")
                if isinstance(row['amenities_list'], list) and len(row['amenities_list']) > 0:
                    st.write("✨ Amenities: " + ", ".join(amenity.replace('_', ' ').title() for amenity in row['amenities_list']))
            
            with col2:
                if "url" in row and not pd.isna(row["url"]):
                    st.markdown(f"[🔗 View Website]({row['url']})")
            
            with col3:
                if st.button("Select this space", key=f"select_{idx}"):
                    st.session_state.selected_space = row["name"]
                    st.success(f"Selected {row['name']}")
            
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
            # Get recommendations based on selected space from tab1
            if 'selected_space' not in st.session_state:
                st.info("Please select a space in the 'Find Spaces' tab first")
            else:
                selected_space = st.session_state.selected_space
                
                # Get the index of the selected space
                selected_idx = recommendation_df[recommendation_df['name'] == selected_space].index[0]
                
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
                    # Use the names to build a list of spaces (selected + recommendations)
                    viz_names = [selected_space] + [rec['name'] for rec in recommendations]
                    # Filter the recommendation dataframe based on these names
                    viz_df = recommendation_df[recommendation_df['name'].isin(viz_names)].copy()
                    # Reorder so the selected space comes first
                    viz_df['order'] = viz_df['name'].apply(lambda x: viz_names.index(x))
                    viz_df = viz_df.sort_values('order').drop('order', axis=1)
                    
                    # Mark the selected space
                    viz_df['type'] = 'Similar Space'
                    viz_df.loc[viz_df['name'] == selected_space, 'type'] = 'Selected Space'
                    
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
    st.subheader("Explore coworking spaces ranked by score and city metrics")
    
    if not has_top_rated:
        st.warning("Top rated spaces data not available")
    else:
        # City filter for top rated tab
        if 'City' in top_rated_df.columns:
            top_rated_cities = sorted(top_rated_df['City'].dropna().unique().tolist())
            selected_top_rated_city = st.selectbox("Select city", top_rated_cities, key="city_filter_tab3")
            
                        # Filter by selected city
            filtered_top_rated = top_rated_df[top_rated_df['City'] == selected_top_rated_city]
            filtered_top_rated = filtered_top_rated.sort_values('Score', ascending=False)
            
            # Show city statistics
            st.subheader(f"Average Prices of the Best Coworkings in {selected_top_rated_city}")
            col1, col2 = st.columns(2)
            
            with col1:
                avg_day_price = filtered_top_rated['Day Pass'].mean()
                if not pd.isna(avg_day_price):
                    st.metric("Average Day Pass", f"${avg_day_price:.2f}")
                else:
                    st.metric("Average Day Pass", "Not available")
            
            with col2:
                avg_month_price = filtered_top_rated['Month Pass'].mean()
                if not pd.isna(avg_month_price):
                    st.metric("Average Monthly Pass", f"${avg_month_price:.2f}")
                else:
                    st.metric("Average Monthly Pass", "Not available")
            
            # Show map
            if 'Latitude' in filtered_top_rated and 'Longitude' in filtered_top_rated:
                st.subheader(f"Map of Coworking Spaces in {selected_top_rated_city}")
                
                map_data = filtered_top_rated[['Latitude', 'Longitude', 'name', 'Score', 'Address']].copy()
                map_data = map_data.dropna(subset=['Latitude', 'Longitude'])
                
                if not map_data.empty:
                    
                    # Create layer with tooltips
                    layer = pdk.Layer(
                        'ScatterplotLayer',
                        data=map_data,
                        get_position=['Longitude', 'Latitude'],
                        get_color=[255, 0, 0, 160],
                        get_radius=100,
                        pickable=True,
                        auto_highlight=True
                    )

                    # Set the viewport location
                    view_state = pdk.ViewState(
                        longitude=map_data['Longitude'].mean(),
                        latitude=map_data['Latitude'].mean(),
                        zoom=12,
                        pitch=0
                    )

                    # Render the deck.gl map with tooltips
                    tooltip = {
                        "html": "<b>Name:</b> {name}<br><b>Score:</b> {Score}<br><b>Address:</b> {Address}",
                        "style": {"background": "white", "color": "black", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"}
                    }

                    r = pdk.Deck(
                        layers=[layer],
                        initial_view_state=view_state,
                        tooltip=tooltip,
                        map_style='mapbox://styles/mapbox/light-v9'
                    )

                    st.pydeck_chart(r)
                else:
                    st.info("No location data available for mapping")
            
            # Display spaces
            st.subheader(f"Coworking Spaces in {selected_top_rated_city}")
            
            for i, (_, row) in enumerate(filtered_top_rated.iterrows()):
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    # Rank display
                    if i == 0:
                        st.markdown(f"<h1 style='text-align: center; color: gold;'>🏆</h1>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='text-align: center;'>#{i+1}</h2>", unsafe_allow_html=True)
                    
                    # Score display
                    if 'Score' in row:
                        st.markdown(f"<h3 style='text-align: center;'>{row['Score']:.1f}⭐</h3>", unsafe_allow_html=True)
                
                with col2:
                    st.subheader(row["name"])
                    
                    # Location info
                    if pd.notna(row["Address"]):
                        st.write(f"📍 {row['Address']}")
                    if pd.notna(row["Neighborhood"]):
                        st.write(f"🏘️ {row['Neighborhood']}")
                    
                    # Ratings info
                    if pd.notna(row["Rating"]):
                        rating_text = f"⭐ {row['Rating']}/5"
                        if pd.notna(row["User Rating Count"]):
                            rating_text += f" ({int(row['User Rating Count'])} reviews)"
                        st.write(rating_text)
                    
                    # Distance from center
                    if pd.notna(row["distance_from_center"]):
                        st.write(f"🚶 {row['distance_from_center']:.2f}km from city center")
                
                st.markdown("---")
        else:
            st.error("City information not available in the dataset")

with tab4:
    st.header("Find Your Perfect Workspace Style")
    
    # Add a friendly explanation of what this tab does
    st.markdown("""
    ### What is this tab about?
    
    This feature groups similar coworking spaces together based on their amenities, prices, and features.
    It helps you discover what type of workspace matches your preferences and working style.
    
    **How it works:**
    - We analyze patterns across all workspaces
    - Similar spaces are grouped into categories
    - You can explore which category fits your needs best
    - See examples of spaces in each category
    """)
    
    # Build clustering model
    n_clusters = 4
    clusters, cluster_centers = build_clustering_model(df, n_clusters)
    
    if clusters is None:
        st.error("Could not build workspace categories with the available data.")
        st.stop()
    
    # Add cluster labels to the dataframe
    df_cluster = df.copy()
    df_cluster['cluster'] = clusters
    
    # Define cluster names and descriptions
    cluster_names = {
        0: "Budget-Friendly Workspaces",
        1: "Premium Full-Service Offices",
        2: "Mid-Range Basic Spaces",
        3: "Specialized Creative Studios"
    }
    
    cluster_descriptions = {
        0: "Affordable spaces with essential amenities. Perfect for freelancers and startups watching their budget.",
        1: "High-end spaces with comprehensive amenities and services. Ideal for established businesses and professionals seeking comfort and convenience.",
        2: "Balanced price-to-amenity ratio. Good for small teams and professionals who need reliable basics without the premium price.",
        3: "Spaces with unique features tailored to specific needs. Great for creative professionals and specialized industries."
    }
    
    # City filter for cluster analysis
    if 'city' in df.columns:
        cities = sorted(df['city'].dropna().unique().tolist())
        selected_city_cluster = st.selectbox(
            "Select a city to explore workspace types", 
            ["All Cities"] + cities, 
            key="city_filter_tab4"
        )

    # Filter data based on selected city
    if selected_city_cluster != "All Cities":
        df_cluster_filtered = df_cluster[df_cluster['city'] == selected_city_cluster]
    else:
        df_cluster_filtered = df_cluster

    # Display filtration info
    st.write(f"Found {len(df_cluster_filtered)} spaces in {selected_city_cluster if selected_city_cluster != 'All Cities' else 'all cities'}")

    if len(df_cluster_filtered) == 0:
        st.warning(f"No coworking spaces found in {selected_city_cluster}")
    else:
        # Create a more visual layout for workspace types
        st.subheader("Workspace Categories")
        st.write("Click on any category to explore its details and see example spaces")
        
        # Show number input for examples
        num_spaces = st.slider("Number of example spaces to show", 
                               min_value=1, max_value=10, value=3)

        # Display each cluster type with its characteristics
        for cluster_id, name in cluster_names.items():
            cluster_data = df_cluster_filtered[df_cluster_filtered['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                with st.expander(f"📍 {name} ({len(cluster_data)} spaces)"):
                    # Add description of this workspace type
                    st.info(cluster_descriptions[cluster_id])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**What you'll typically find:**")
                        amenity_freqs = {}
                        amenity_cols = [col for col in df_cluster.columns if col.startswith('has_amenity_')]
                        for col in amenity_cols:
                            amenity_name = col.replace('has_amenity_', '').replace('_', ' ').title()
                            freq = cluster_data[col].mean()
                            if freq > 0.3:  # Only show significant amenities
                                amenity_freqs[amenity_name] = freq
                        
                        # Show top 5 amenities with percentages and visual indicators
                        for amenity, freq in sorted(amenity_freqs.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True)[:5]:
                            percentage = int(freq * 100)
                            # Create a simple visual bar
                            bar_length = int(percentage / 5)  # Scale to reasonable length
                            bar = "🟩" * bar_length
                            st.write(f"✨ {amenity}: {bar} ({percentage}%)")
                    
                    with col2:
                        avg_price = cluster_data['price_numeric'].mean()
                        if not pd.isna(avg_price):
                            st.write("**Price range:**")
                            price_min = cluster_data['price_numeric'].min()
                            price_max = cluster_data['price_numeric'].max()
                            st.write(f"💰 Avg: {avg_price:.2f} (Range: {price_min:.2f} - {price_max:.2f})")
                        
                        st.write("**Availability:**")
                        st.write(f"🏢 {len(cluster_data)} spaces in {selected_city_cluster if selected_city_cluster != 'All Cities' else 'selected cities'}")
                        
                        # Add a "best for" section
                        st.write("**Best for:**")
                        if cluster_id == 0:
                            st.write("👩‍💻 Freelancers and startups on a budget")
                        elif cluster_id == 1:
                            st.write("👔 Established businesses and professionals")
                        elif cluster_id == 2:
                            st.write("👥 Small teams and independent professionals")
                        else:
                            st.write("🎨 Creative professionals and specialized industries")

                    # Show example spaces button
                    if st.button(f"Show me spaces in this category", key=f"show_spaces_{cluster_id}"):
                        matching_spaces = cluster_data.head(num_spaces)
                        st.write(f"**Example spaces in this category:**")
                        
                        for _, space in matching_spaces.iterrows():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.subheader(space["name"])
                                if "address" in space and not pd.isna(space["address"]):
                                    st.write(f"📍 {space['address']}")
                                if "price" in space and not pd.isna(space["price"]):
                                    st.write(f"💰 {space['price']}")
                                if isinstance(space['amenities_list'], list):
                                    st.write("✨ " + ", ".join(
                                        amenity.replace('_', ' ').title() 
                                        for amenity in space['amenities_list'][:5]
                                    ))
                            
                            with col2:
                                if "url" in space and not pd.isna(space["url"]):
                                    st.markdown(f"[🔗 View Website]({space['url']})")
                            st.markdown("---")
                            
with tab5:
    with tab5:
    st.header("🤖 AI-Powered Coworking Scoring System")
    
    st.markdown("""
    ### What does this system do?
    
    This AI system learns from existing coworking spaces to:
    - **Predict quality scores** for new coworking spaces (1-5 scale)
    - **Analyze factors** that contribute to coworking quality
    - **Provide intelligent recommendations** using machine learning
    
    The system considers amenities, pricing, location, and other factors to make predictions.
    """)
    
    # Train the model
    if st.button("🚀 Train AI Model", type="primary"):
        with st.spinner("Training AI model... This may take a moment"):
            try:
                model, feature_columns, metrics = train_scoring_model(df)
                
                if model is None:
                    st.error(f"Could not train model: {metrics.get('error', 'Unknown error')}")
                else:
                    st.session_state.scoring_model = model
                    st.session_state.feature_columns = feature_columns
                    st.session_state.model_metrics = metrics
                    
                    st.success("✅ Model trained successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R² Score", f"{metrics['r2']:.3f}")
                    with col2:
                        st.metric("Mean Squared Error", f"{metrics['mse']:.3f}")
                    
                    st.info("Now you can use the prediction system below!")
                    
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    # Show model info if trained
    if 'scoring_model' in st.session_state:
        st.markdown("---")
        st.subheader("🎯 Predict Score for New Coworking Space")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📍 Basic Information**")
            
            # City selection
            available_cities = df['city'].dropna().unique() if 'city' in df.columns else ['Madrid', 'Barcelona', 'Valencia']
            new_city = st.selectbox("City", available_cities, key="scoring_city")
            
            # Price input
            new_price = st.number_input(
                "Monthly Price (€)", 
                min_value=0, 
                max_value=2000, 
                value=300,
                step=50,
                key="scoring_price"
            )
            
            # Total amenities
            total_amenities = st.number_input(
                "Total number of amenities", 
                min_value=0, 
                max_value=20, 
                value=5,
                key="scoring_amenities"
            )
        
        with col2:
            st.markdown("**✨ Key Amenities**")
            
            # Get top amenities for selection
            top_amenities = [amenity for amenity, _ in most_common_amenities[:8]]
            
            # Create checkboxes for amenities
            selected_amenities = {}
            for i, amenity in enumerate(top_amenities):
                amenity_display = amenity.replace('_', ' ').title()
                selected_amenities[amenity] = st.checkbox(
                    amenity_display, 
                    key=f"scoring_amenity_{i}"
                )
        
        # Prediction button
        if st.button("🔮 Predict Quality Score", type="secondary"):
            try:
                # Prepare parameters
                new_params = {
                    'total_amenities': total_amenities,
                    'price_normalized': (new_price - 100) / (1000 - 100) if new_price > 0 else 0.5,  # Simple normalization
                    'city_encoded': hash(new_city) % 10,  # Simple city encoding
                }
                
                # Add amenity flags
                for amenity, is_selected in selected_amenities.items():
                    new_params[f'has_amenity_{amenity}'] = 1 if is_selected else 0
                
                # Predict score
                predicted_score = predict_coworking_score(
                    st.session_state.scoring_model,
                    st.session_state.feature_columns,
                    new_params
                )
                
                # Display result with nice formatting
                st.markdown("---")
                
                # Score display
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f8f9fa;">
                        <h2 style="color: #4CAF50; margin: 0;">🎯 Predicted Score</h2>
                        <h1 style="color: #2E7D32; margin: 10px 0;">{predicted_score:.1f}/5.0</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Interpretation
                if predicted_score >= 4.0:
                    st.success("🌟 **Excellent** - This would be a top-tier coworking space!")
                    interpretation = "This space has all the factors that make coworking spaces successful!"
                elif predicted_score >= 3.5:
                    st.success("👍 **Very Good** - This would be a solid choice!")
                    interpretation = "This space offers great value and good amenities."
                elif predicted_score >= 3.0:
                    st.warning("👌 **Good** - This would be a decent option!")
                    interpretation = "This space meets basic requirements but could be improved."
                elif predicted_score >= 2.5:
                    st.warning("⚠️ **Fair** - This might have some limitations")
                    interpretation = "Consider adding more amenities or adjusting pricing."
                else:
                    st.error("❌ **Needs Improvement** - Consider major changes")
                    interpretation = "This space needs significant improvements to be competitive."
                
                st.info(f"💡 **AI Analysis**: {interpretation}")
                
                # Show feature importance
                st.markdown("---")
                st.subheader("📊 What Influences This Score?")
                
                # Simple feature importance display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Your Input:**")
                    st.write(f"• **City**: {new_city}")
                    st.write(f"• **Monthly Price**: €{new_price}")
                    st.write(f"• **Total Amenities**: {total_amenities}")
                    
                with col2:
                    st.markdown("**Selected Amenities:**")
                    selected_list = [amenity.replace('_', ' ').title() for amenity, selected in selected_amenities.items() if selected]
                    if selected_list:
                        for amenity in selected_list:
                            st.write(f"✅ {amenity}")
                    else:
                        st.write("No specific amenities selected")
                
                # Suggestions for improvement
                if predicted_score < 4.0:
                    st.markdown("---")
                    st.subheader("💡 Suggestions for Improvement")
                    
                    suggestions = []
                    if total_amenities < 8:
                        suggestions.append("Consider adding more amenities to increase appeal")
                    if new_price > 500:
                        suggestions.append("Price might be high for the market - consider competitive pricing")
                    if sum(selected_amenities.values()) < 3:
                        suggestions.append("Adding key amenities like coffee, meeting rooms, or parking could help")
                    
                    if suggestions:
                        for suggestion in suggestions:
                            st.write(f"• {suggestion}")
                    else:
                        st.write("• Focus on enhancing unique features that set your space apart")
                        st.write("• Consider location and accessibility improvements")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        
        # Model information
        st.markdown("---")
        with st.expander("🔍 Model Information"):
            st.write("**Model Type**: Random Forest Regressor")
            st.write(f"**Number of Features**: {len(st.session_state.feature_columns)}")
            st.write(f"**Training Data**: {len(df)} coworking spaces")
            st.write("**Features Used**:")
            for feature in st.session_state.feature_columns[:10]:  # Show first 10
                feature_display = feature.replace('has_amenity_', '').replace('_', ' ').title()
                st.write(f"  • {feature_display}")
            if len(st.session_state.feature_columns) > 10:
                st.write(f"  • ... and {len(st.session_state.feature_columns) - 10} more")
    
    else:
        # Instructions when model is not trained
        st.info("👆 **Get started**: Click 'Train AI Model' above to begin using the scoring system!")
        
        st.markdown("---")
        st.subheader("📚 How it works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🧠 Machine Learning Process:**
            1. Analyzes existing coworking data
            2. Learns patterns from amenities, prices, locations
            3. Creates a predictive model
            4. Scores new spaces based on learned patterns
            """)
        
        with col2:
            st.markdown("""
            **🎯 What you can predict:**
            - Quality score (1-5 scale) for new spaces
            - Impact of different amenities
            - Optimal pricing strategies
            - Location effectiveness
            """)
        
        st.markdown("---")
        st.markdown("""
        **💡 Use cases:**
        - **Entrepreneurs**: Validate your coworking space concept
        - **Investors**: Assess potential locations and setups
        - **Managers**: Optimize existing space offerings
        - **Researchers**: Understand coworking space success factors
        """)