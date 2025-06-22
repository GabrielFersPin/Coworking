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

def normalize_price_by_country(df):
    """
    Normalize prices within each country/region to handle different currencies
    """
    # Create a copy to avoid modifying original data
    df_normalized = df.copy()
    
    # Create a normalized price column
    df_normalized['price_normalized'] = np.nan
    
    # Group by country/city and normalize within each group
    if 'country' in df.columns:
        # If we have country information, normalize by country
        for country in df['country'].unique():
            country_mask = df['country'] == country
            country_prices = df.loc[country_mask, 'price_numeric']
            
            if len(country_prices) > 0:
                price_min = country_prices.min()
                price_max = country_prices.max()
                
                if price_max > price_min:
                    df_normalized.loc[country_mask, 'price_normalized'] = (
                        (country_prices - price_min) / (price_max - price_min)
                    )
    else:
        # If no country info, normalize by city
        for city in df['city'].unique():
            city_mask = df['city'] == city
            city_prices = df.loc[city_mask, 'price_numeric']
            
            if len(city_prices) > 0:
                price_min = city_prices.min()
                price_max = city_prices.max()
                
                if price_max > price_min:
                    df_normalized.loc[city_mask, 'price_normalized'] = (
                        (city_prices - price_min) / (price_max - price_min)
                    )
    
    # Fill any remaining NaN values with median
    df_normalized['price_normalized'] = df_normalized['price_normalized'].fillna(0.5)
    
    return df_normalized['price_normalized']

# Update the create_features_for_scoring function
def create_features_for_scoring(df):
    """
    Create numerical features for the scoring model with improved price handling
    """
    features_df = df.copy()
    
    # 1. Amenity features (existing code)
    amenity_cols = [col for col in df.columns if col.startswith('has_amenity_')]
    
    # 2. Improved price normalization
    if 'price_numeric' in df.columns:
        features_df['price_normalized'] = normalize_price_by_country(df)
    else:
        features_df['price_normalized'] = 0.5
    
    # 3. City encoding (existing code)
    if 'city' in df.columns:
        le_city = LabelEncoder()
        city_data = df['city'].fillna('Unknown')
        features_df['city_encoded'] = le_city.fit_transform(city_data)
    else:
        features_df['city_encoded'] = 0
    
    # 4. Count total amenities (existing code)
    if amenity_cols:
        features_df['total_amenities'] = features_df[amenity_cols].sum(axis=1)
    else:
        features_df['total_amenities'] = 0
    
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

def get_price_analysis(space_data, df):
    """
    Get price analysis for a space, comparing only with spaces in the same city/country.
    Returns: (comparison, price, avg_price, scope) or (None, None, None, None) if no comparison possible
    """
    if 'price_numeric' not in space_data or pd.isna(space_data['price_numeric']):
        return None, None, None, None

    # First try to compare within the same city
    if 'city' in space_data and not pd.isna(space_data['city']):
        city_mask = (df['city'] == space_data['city']) & (df['price_numeric'].notna())
        if 'is_invalid_price' in df.columns:
            city_mask = city_mask & (~df['is_invalid_price'])
        city_spaces = df[city_mask]
        if len(city_spaces) > 1:
            city_avg = city_spaces['price_numeric'].mean()
            comparison = "above" if space_data['price_numeric'] > city_avg else "below"
            return comparison, space_data['price_numeric'], city_avg, "city"

    # If no city comparison possible, try country comparison
    if 'country' in space_data and not pd.isna(space_data['country']):
        country_mask = (df['country'] == space_data['country']) & (df['price_numeric'].notna())
        if 'is_invalid_price' in df.columns:
            country_mask = country_mask & (~df['is_invalid_price'])
        country_spaces = df[country_mask]
        if len(country_spaces) > 1:
            country_avg = country_spaces['price_numeric'].mean()
            comparison = "above" if space_data['price_numeric'] > country_avg else "below"
            return comparison, space_data['price_numeric'], country_avg, "country"

    return None, None, None, None

# App title and description
st.title("Coworking Space Explorer & AI Analyzer")
st.write("""
Discover, compare, and analyze coworking spaces with the power of AI.
- 🔍 **Filter** by city, price, and amenities
- 🤖 **Get AI-powered quality scores** and recommendations
- 📊 **Compare spaces** and explore workspace styles
- 🏆 **Find top-rated and best-matching coworkings** for your needs

Start by selecting your preferences or jump to the AI tab for a detailed analysis!
""")

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
    st.header("🤖 AI-Powered Coworking Scoring System")
    st.markdown("""
    Welcome to the AI Scoring tab!  
    Here you can use artificial intelligence to **evaluate coworking spaces** based on their amenities, price, and location.

    **How to use this tab:**
    1. Select a coworking space you want to analyze. The AI will predict its quality score and show a detailed breakdown.
    2. Use the recommendations and analysis to improve your space or compare options.

    _Scroll down to get started!_
    """)

    # --- Place the button here ---
    if st.button("Train & Reload AI Model"):
        with st.spinner("Training and exporting model..."):
            model, feature_columns, metrics = train_scoring_model(df)
            if model is not None:
                os.makedirs("src/ai", exist_ok=True)
                joblib.dump(model, "src/ai/scoring_model.pkl")
                joblib.dump(feature_columns, "src/ai/feature_columns.pkl")
                # Reload the model into session state immediately
                st.session_state.scoring_model = model
                st.session_state.feature_columns = feature_columns
                st.success("Model trained, exported, and reloaded successfully!")
                st.write(f"Model metrics: {metrics}")
            else:
                st.error("Model training failed. Check your data.")
    # --- End button placement ---

    st.subheader("📊 AI Analysis of Existing Coworking Spaces")

    # Check if model is trained
    if 'scoring_model' not in st.session_state:
        st.info("⚠️ The AI model must be trained in the backend before using this feature.")
    else:
        st.success("✅ AI Model is ready!")

        # Space selection
        available_spaces = df['name'].dropna().unique() if 'name' in df.columns else []

        if len(available_spaces) == 0:
            st.error("No coworking spaces available for scoring.")
        else:
            # If a space is selected in tab1, use it directly and don't show selectbox
            if 'selected_space' in st.session_state and st.session_state.selected_space in available_spaces:
                selected_space_name = st.session_state.selected_space
                st.success(f"Using space selected in 'Find Spaces': **{selected_space_name}**")
            else:
                selected_space_name = st.selectbox(
                    "Select a coworking space to analyze:",
                    available_spaces,
                    key="existing_space_scorer"
                )

            if st.button("🔍 Analyze This Space", type="primary"):
                try:
                    # Get space data
                    space_data = df[df['name'] == selected_space_name].iloc[0]
                    
                    # Prepare features for scoring
                    features_df = create_features_for_scoring(df)
                    space_features = features_df[features_df.index == space_data.name].iloc[0]
                    
                    # Create parameters for prediction
                    space_params = {}
                    for col in st.session_state.feature_columns:
                        if col in space_features:
                            space_params[col] = space_features[col]
                        else:
                            space_params[col] = 0
                    
                    # Predict score
                    predicted_score = predict_coworking_score(
                        st.session_state.scoring_model,
                        st.session_state.feature_columns,
                        space_params
                    )
                    
                    # Display comprehensive analysis
                    st.markdown("---")
                    
                    # Header with space info
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.subheader(f"📋 Analysis: {selected_space_name}")
                        if 'city' in space_data and not pd.isna(space_data['city']):
                            st.write(f"🏙️ **Location**: {space_data['city']}")
                        if 'address' in space_data and not pd.isna(space_data['address']):
                            st.write(f"📍 **Address**: {space_data['address']}")
                    
                    with col2:
                        if 'url' in space_data and not pd.isna(space_data['url']):
                            st.markdown(f"[🔗 Visit Website]({space_data['url']})")
                    
                    # Score display with detailed breakdown
                    st.markdown("### 🎯 AI Quality Score")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        # Score with color coding
                        if predicted_score >= 4.0:
                            color = "#4CAF50"
                            emoji = "🌟"
                            rating = "Excellent"
                        elif predicted_score >= 3.5:
                            color = "#8BC34A"
                            emoji = "👍"
                            rating = "Very Good"
                        elif predicted_score >= 3.0:
                            color = "#FFC107"
                            emoji = "👌"
                            rating = "Good"
                        elif predicted_score >= 2.5:
                            color = "#FF9800"
                            emoji = "⚠️"
                            rating = "Fair"
                        else:
                            color = "#F44336"
                            emoji = "❌"
                            rating = "Needs Improvement"
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 15px; background-color: #f8f9fa;">
                            <h3 style="color: {color}; margin: 0;">{emoji} {rating}</h3>
                            <h1 style="color: {color}; margin: 10px 0; font-size: 3em;">{predicted_score:.1f}</h1>
                            <p style="color: #666; margin: 0;">out of 5.0</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed analysis sections
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 💰 Pricing Analysis")
                        if 'price' in space_data and not pd.isna(space_data['price']):
                            st.write(f"**Listed Price**: {space_data['price']}")
                        
                        if 'price_numeric' in space_data and not pd.isna(space_data['price_numeric']):
                            comparison, price, avg_price, scope = get_price_analysis(space_data, df)
                            if comparison and avg_price:
                                st.write(f"**Price Analysis**: {comparison} {scope} average (€{price:.0f} vs €{avg_price:.0f} avg)")
                            else:
                                st.write(f"**Price**: €{space_data['price_numeric']:.0f}")
                                st.write("(Not enough data in the area for price comparison)")
                    
                    with col2:
                        st.markdown("### 🏙️ Location Insights")
                        if 'city' in space_data and not pd.isna(space_data['city']):
                            city_spaces = df[df['city'] == space_data['city']]
                            st.write(f"**City**: {space_data['city']}")
                            st.write(f"**Competition**: {len(city_spaces)} spaces in this city")
                    
                    # Amenities analysis
                    st.markdown("---")
                    st.markdown("### ✨ Amenities Analysis")
                    if isinstance(space_data['amenities_list'], list):
                        total_amenities = len(space_data['amenities_list'])
                        avg_amenities = df['amenities_list'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
                        
                        st.write(f"**Total Amenities**: {total_amenities}")
                        st.write(f"**Compared to Average**: {total_amenities - avg_amenities:.1f} amenities")
                        
                        if total_amenities > 0:
                            st.write("**Available Amenities**:")
                            for amenity in space_data['amenities_list'][:8]:  # Show first 8
                                st.write(f"  ✅ {amenity.replace('_', ' ').title()}")
                            if len(space_data['amenities_list']) > 8:
                                st.write(f"  ... and {len(space_data['amenities_list']) - 8} more")
                    
                    # Competitive analysis
                    st.markdown("---")
                    st.markdown("### 📈 Competitive Position")
                    
                    # Compare with similar spaces in the same city
                    if 'city' in space_data and not pd.isna(space_data['city']):
                        city_spaces = df[df['city'] == space_data['city']]
                        
                        # Score all spaces in the same city
                        city_scores = []
                        for _, other_space in city_spaces.iterrows():
                            try:
                                other_features = features_df[features_df.index == other_space.name].iloc[0]
                                other_params = {}
                                for col in st.session_state.feature_columns:
                                    if col in other_features:
                                        other_params[col] = other_features[col]
                                    else:
                                        other_params[col] = 0
                                
                                other_score = predict_coworking_score(
                                    st.session_state.scoring_model,
                                    st.session_state.feature_columns,
                                    other_params
                                )
                                city_scores.append(other_score)
                            except:
                                city_scores.append(3.0)  # Default score if calculation fails
                        
                        if city_scores:
                            avg_city_score = np.mean(city_scores)
                            rank = sum(1 for score in city_scores if score > predicted_score) + 1
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("City Average Score", f"{avg_city_score:.1f}")
                            with col2:
                                st.metric("Your Rank", f"#{rank} of {len(city_scores)}")
                            with col3:
                                percentile = ((len(city_scores) - rank + 1) / len(city_scores)) * 100
                                st.metric("Percentile", f"{percentile:.0f}%")
                    
                    # Recommendations
                    st.markdown("---")
                    st.markdown("### 💡 AI Recommendations")
                    
                    recommendations = []
                    
                    # Amenity recommendations
                    if isinstance(space_data['amenities_list'], list):
                        current_amenities = set(space_data['amenities_list'])
                        popular_amenities = [amenity for amenity, _ in most_common_amenities[:10]]
                        missing_popular = [amenity for amenity in popular_amenities if amenity not in current_amenities]
                        
                        if missing_popular[:3]:  # Show top 3 missing popular amenities
                            recommendations.append(f"**Add Popular Amenities**: Consider adding {', '.join(missing_popular[:3]).replace('_', ' ').title()}")
                    
                    # Price recommendations
                    if 'price_numeric' in space_data and not pd.isna(space_data['price_numeric']):
                        avg_price = df['price_numeric'].mean()
                        if space_data['price_numeric'] > avg_price * 1.2:
                            recommendations.append("**Pricing Strategy**: Consider if premium pricing is justified by unique amenities")
                        elif space_data['price_numeric'] < avg_price * 0.8:
                            recommendations.append("**Value Optimization**: You could potentially increase pricing or add premium services")
                    
                    # Score-based recommendations
                    if predicted_score < 3.5:
                        recommendations.append("**Focus Areas**: Enhance key amenities and consider competitive positioning")
                    elif predicted_score >= 4.0:
                        recommendations.append("**Maintain Excellence**: Focus on maintaining current quality and unique differentiators")
                    
                    # Display recommendations
                    if recommendations:
                        for rec in recommendations:
                            st.write(f"• {rec}")
                    else:
                        st.write("• This space appears well-positioned in the market")
                        st.write("• Continue monitoring competitor offerings and customer feedback")
                    
                    # Description analysis if available
                    if 'description' in space_data and not pd.isna(space_data['description']):
                        st.markdown("---")
                        with st.expander("📝 Space Description"):
                            st.write(space_data['description'])
                
                except Exception as e:
                    st.error(f"Error analyzing space: {str(e)}")
                    st.write("Debug info:", str(e))

# Auto-load the AI model and feature columns if available
model_path = os.path.join(current_dir, "src", "ai", "scoring_model.pkl")
features_path = os.path.join(current_dir, "src", "ai", "feature_columns.pkl")

if 'scoring_model' not in st.session_state or 'feature_columns' not in st.session_state:
    try:
        st.session_state.scoring_model = joblib.load(model_path)
        st.session_state.feature_columns = joblib.load(features_path)
    except Exception as e:
        st.warning(f"AI model could not be loaded automatically: {e}")
