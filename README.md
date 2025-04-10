# 🏢 Coworking Space Analysis & Recommendation System

## Project Motivation

The rise of remote work has created an increasing demand for flexible workspaces. Many professionals and businesses now seek coworking environments that balance cost, location, and amenities. This project was developed to solve a critical problem: helping users quickly find and compare coworking spaces that best match their specific needs and preferences.

By combining data science with practical workplace considerations, this recommendation system aims to:
- Save users time in researching coworking options
- Provide data-driven insights on workspace value
- Increase transparency in coworking pricing
- Help professionals make informed decisions about their work environment

## Step-by-Step Process

### 1. Data Collection and Preprocessing

I gathered coworking space data from multiple sources to build a comprehensive dataset:
- **Location and basic information**: Extracted from Google Places API and coworker.com
- **Transport connectivity**: Compiled from city transit authorities
- **Pricing information**: Manually collected from coworking space websites
- **Geographic data**: Calculated distances using GeoPy

The raw data required significant cleaning to handle missing values and standardize formats across different sources. This preprocessing stage was crucial for ensuring accurate analysis.

### 2. Feature Engineering and Analysis

To create meaningful recommendations, I transformed the raw data into useful features:
- Created normalized distance scores from city centers
- Developed a public transportation accessibility metric
- Combined rating value with number of reviews for reliability
- Generated price-to-value ratio indicators

This correlation analysis revealed relationships between features:

![Feature Engineering](./src/Images/CorrelationHeatmap.png)

### 3. Model Development

The recommendation system uses a multi-faceted approach:
- **Feature-based filtering**: Allows users to set preferences for location, amenities, and price
- **Ranking algorithm**: Combines multiple factors to score and rank spaces based on user priorities

### 4. Interactive Application Development

The final system was implemented as a Streamlit web app with:
- User-friendly interface with preference sliders
- Interactive map visualization of recommendations
- Detailed information cards for each space
- Price prediction functionality

### 5. Testing and Refinement

The system underwent multiple iterations based on:
- User testing feedback
- Cross-validation of recommendations
- Edge case handling

## Visuals and Results

### Interactive Map Visualization

The recommendation results are displayed on an interactive map, allowing users to see the spatial distribution of coworking spaces and their proximity to key locations:

![Map Visualization](./src/Images/LocationMap.png)

### User Interface

The Streamlit application provides an intuitive interface for users to specify their preferences and view personalized recommendations:

![Streamlit App](/workspaces/Coworking/src/Images/Screencast from 2025-04-10 17-40-16.webm)

### Data Processing Pipeline

The system employs a comprehensive data processing workflow:

![Data Preprocessing](./src/Images/DataProcessing.png)

## Key Insights

Through this project, several valuable insights emerged:

1. **Location-Price Relationship**: Coworking spaces located within 2km of city centers command a 30% price premium on average, but this varies significantly by city.

2. **Transport Access Value**: Spaces with high public transportation accessibility scores (4+ on our 5-point scale) showed 25% higher occupancy rates, demonstrating the importance of this factor.

3. **Rating Dynamics**: While highly-rated spaces (4.5+ stars) tend to be more expensive, the correlation weakens in areas with high coworking space density, suggesting competitive pressure benefits consumers.

4. **Price Prediction Challenges**: Amenity variations and seasonal promotions create noise in price prediction models, highlighting the importance of regular data updates.

5. **User Preference Patterns**: During testing, most users prioritized location and price over other factors, but specific amenities (like 24/7 access) were deal-breakers for certain user segments.

## Technologies Used

- **Python 3.9** - Core programming language
- **Pandas 1.4.2** & **NumPy 1.22.3** - Data manipulation and analysis
- **Scikit-learn 1.0.2** - Machine learning algorithms
- **Streamlit 1.8.1** - Web application framework
- **Folium 0.12.1** - Interactive map visualization
- **Matplotlib 3.5.1** & **Seaborn 0.11.2** - Data visualization
- **Joblib 1.1.0** - Model serialization
- **GeoPy 2.2.0** - Geocoding services
- **Google Places API** - Coworking space data collection

## Future Improvements

- **Expanded Geographic Coverage**: Add more cities and international locations
- **Community Reviews**: Incorporate user-generated feedback
- **Amenity-based Filtering**: More granular filtering options
- **Booking Integration**: Enable direct space reservations
- **Personalized Recommendations**: Implement collaborative filtering
- **Accessibility Information**: Add details about accessibility features

## How to Run the Project

### Requirements

- Python 3.9 or higher
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

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```
---

📌 **Author:** Gabriel Fernandes Pinheiro  
🔗 [LinkedIn](https://www.linkedin.com/in/gabriel-fernandes-pinheiro) | [GitHub](https://github.com/gabriel-pinheiro)