# 🏢 Coworking Space Analysis & Recommendation System

## Project Motivation

The rise of remote work has created an increasing demand for flexible workspaces. Many professionals and businesses now seek coworking environments that balance cost, location, and amenities. This project was developed to solve a critical problem: helping users quickly find and compare coworking spaces that best match their specific needs and preferences.

By combining data science with practical workplace considerations, this recommendation system aims to:
- Save users time in researching coworking options
- Provide data-driven insights on workspace value
- Increase transparency in coworking pricing
- Help professionals make informed decisions about their work environment

## Features

### 1. Find Spaces
- **City-based Filtering**: Narrow down options by selecting specific cities
- **Amenity Selection**: Choose must-have amenities from the most common options
- **Price Range Filtering**: Set budget constraints with simplified price brackets
- **Detailed Results**: View comprehensive information including location, pricing, and available amenities

### 2. Similar Spaces
- **Recommendation Engine**: Find similar workspaces based on selected coworking space
- **Similarity Visualization**: Interactive scatter plots showing price vs. amenity count
- **Amenity Comparison**: Visual heatmap showing amenity differences between similar spaces
- **Customizable Results**: Adjust number of recommendations to see more or fewer options

### 3. Top Rated Spaces
- **City Statistics**: View average prices for day passes and monthly passes
- **Interactive Map**: Explore geographic distribution of top-rated spaces
- **Ranking System**: Spaces ranked by overall score with distance from center metrics
- **Rich Details**: Information on ratings, reviews, neighborhoods, and addresses

### 4. Cluster Analysis
- **Automated Clustering**: Discover natural groupings of similar coworking spaces
- **Elbow Method**: Scientifically determine optimal number of clusters or set manually
- **Cluster Characteristics**: Identify defining features of each space cluster
- **Visual Analysis**: Heatmaps and scatter plots showing amenity distribution across clusters

## Data Processing & Analysis

### Data Collection and Preprocessing

The system processes several datasets to provide comprehensive analysis:
- **Amenity data**: Extracted from descriptions and structured data sources
- **Coworking space details**: Location, pricing, and basic information
- **Rating and distance metrics**: For top-rated space rankings

The data preprocessing pipeline includes:
- Cleaning price information and removing invalid sequential prices
- Converting string representations of amenities into structured lists
- Enhancing amenity detection through description text analysis
- Normalizing features for similarity calculations

### Recommendation System

The recommendation engine uses:
- **Cosine similarity**: To find spaces with similar amenity profiles and feature sets
- **Feature-based filtering**: Allows precise filtering by city, amenities, and price range
- **Price normalization**: Ensures fair comparison across different price points

### Clustering Algorithm

The cluster analysis employs:
- **K-means clustering**: Groups similar spaces based on amenities and features
- **Elbow method**: Determines optimal cluster count through sum of squared distances
- **Outlier handling**: Uses IQR method to remove price outliers before clustering
- **Feature standardization**: Normalizes price data for fair comparison

## Visuals and Results

### Interactive Map Visualization

The recommendation results are displayed on an interactive map, allowing users to see the spatial distribution of coworking spaces:

![Map Visualization](./src/Images/LocationMap.png)

### Similarity Analysis

Users can visualize relationships between similar spaces:

![Feature Engineering](./src/Images/CorrelationHeatmap.png)

### Cluster Insights

The cluster analysis provides insights into natural groupings of coworking spaces:

![Data Preprocessing](./src/Images/DataProcessing.png)

## Key Insights

Through this project, several valuable insights emerged:

1. **Amenity Importance**: Certain amenities like WiFi, coffee, and meeting rooms are nearly universal, while others like bike storage and 24/7 access are key differentiators.

2. **Price-Amenity Relationship**: Spaces with more amenities generally command higher prices, but the correlation varies significantly by city and neighborhood.

3. **Clustering Patterns**: Coworking spaces naturally cluster into distinct groups based on their amenity profiles and price points.

4. **Similarity Metrics**: Spaces that appear different at first glance often have similar feature profiles when analyzed quantitatively.

5. **Data Quality Challenges**: Price information varies in format and reliability, requiring robust cleaning and validation processes.

## Technologies Used

- **Python** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Streamlit** - Interactive web application framework
- **Scikit-learn** - Machine learning algorithms for clustering and similarity
- **Matplotlib & Seaborn** - Static data visualization
- **Plotly Express** - Interactive data visualization
- **PyDeck** - Geographic visualization
- **Regular Expressions** - Pattern matching for data extraction

## How to Run the Project

### Requirements

- Python 3.7 or higher
- Required packages listed in `requirements.txt`

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/gabriel-pinheiro/coworking-space-recommendation.git
   cd coworking-space-recommendation
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure data files are in the correct location:
   ```
   src/results/extracted_amenities.csv
   src/results/merged_coworking_spaces.csv
   src/results/MergedPlacesScoreDistance.csv (optional)
   ```

5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## Future Improvements

- **User Preferences**: Save user preferences for faster future searches
- **Additional Data Sources**: Integrate more coworking space databases
- **Advanced Filtering**: Add more granular filtering options like noise level and workspace type
- **Mobile Optimization**: Improve responsive design for mobile users
- **Community Reviews**: Incorporate user-generated feedback
- **Booking Integration**: Enable direct space reservations

---

📌 **Author:** Gabriel Fernandes Pinheiro  
🔗 [LinkedIn](https://www.linkedin.com/in/gabriel-fernandes-pinheiro) | [GitHub](https://github.com/gabriel-pinheiro)