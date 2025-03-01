# 🏢 Coworking Space Analysis & Recommendation System

## Coworking Space Recommendation System

## Project Overview

This project focuses on creating a recommendation system for coworking spaces based on various features such as location, rating, transport facilities, and distance from the center. It includes a comprehensive analysis and prediction system to help users find the best coworking spaces in their preferred city.

## Features

- Coworking space recommendations based on user preferences (rating, distance, transport, etc.).
- Visualization of recommended coworking spaces on an interactive map.
- Predicted prices for coworking space daily passes.
- Interactive Streamlit web app for seamless user experience.

## Data Sources

The data for this project was gathered from multiple sources, including:

- Coworking space information (location, rating, number of reviews).
- External APIs and manual data collection for additional insights.

## Process

### 1. Data Collection and Preprocessing

The first step was to gather data from various sources, such as Google Maps and coworking space directories.

![Data Preprocessing](/workspaces/Coworking/src/Images/DataProcessing.png)

### 2. Feature Engineering and Analysis

Data transformation and feature extraction were performed to make sure we had useful features for the recommendation model. Below is a table that shows the most important features used in the model:

![Feature Engineering](/workspaces/Coworking/src/Images/CorrelationHeatmap.png)

### 3. Model Training and Prediction

The recommendation system was built using a machine learning model.

### 4. Map Visualization of Recommended Spaces

Once the recommendations were generated, we used Folium to display the recommended coworking spaces on an interactive map:

![Map Visualization](/workspaces/Coworking/src/Images/LocationMap.png)

### 5 Challenges and solutions:

- The problems with the Google Maps API: Th API just give me 5 reviews per request, so I could not do the properly sentimental analysis and study the amenities and necessities from the clients.
- Find online information: I couldn't' find enough information online about the coworking spaces, API's or complete websites.
- Apply the predictions on the Streamlit: It was very challenging to use the user input into the prediction model to predict the price.

### 6 Future Improvements:

- Expanded Geographic Coverage: Add more cities and international locations to the database
- Community Reviews: Incorporate user-generated reviews and ratings
- Amenity-based Filtering: Allow users to filter spaces based on specific amenities (standing desks, private rooms, etc.)
- Price Prediction Refinement: Improve the price prediction algorithm with more historical data
- Booking Integration: Enable direct booking of coworking spaces through the platform
- Personalized Recommendations: Implement collaborative filtering to suggest spaces based on similar users' preferences
- Accessibility Information: Add detailed information about accessibility features of each space

### 7. Deployment and Streamlit App

The final recommendation system was deployed using Streamlit, making it easy for users to input their preferences and view the recommendations directly on the web.

Here’s a screenshot of the deployed app interface:

![Streamlit App](/workspaces/Coworking/src/Images/StreamlitScreenshot.png)

## How to Run the Project

### Requirements

- Python 3.x
- Streamlit
- Pandas
- Scikit-learn
- Folium
- Joblib
- Other necessary libraries (listed in `requirements.txt`)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/coworking-space-recommendation.git
  

---

 **Author:** Gabriel Fernandes Pinheiro  
🔗 [LinkedIn](https://www.linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
