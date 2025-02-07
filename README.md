# Coworking Space Market Analysis

This project analyzes the coworking space market to understand the main factors influencing prices and identify the best features to offer in a new coworking space. The analysis uses data from Google Maps reviews, API extractions, and demographic information, along with machine learning techniques to predict prices and visualize patterns.

## Project Overview

The primary goal of this project is to provide insights into the coworking space market, focusing on:

- **Price Prediction:** Using regression models to predict prices of coworking spaces based on neighborhood features.
- **Text Analysis:** Analyzing Google Maps reviews to extract keywords and identify important features of top coworking spaces.
- **Data Visualization:** Visualizing trends and correlations in the data using charts and word clouds.

## Datasets Used

1. **API Extraction Data**: Data from Google Maps reviews, including ratings and feedback on coworking spaces.
2. **Top Coworking Spaces**: Reviews from the top two coworking spaces, used to extract keywords and their importance.
3. **Demographic and Location Data**: Information about neighborhoods and coworking spaces, such as population, unemployment rates, and available amenities.

The dataset for demographic and coworking space features includes the following columns:

- **Neighborhood**: The neighborhood of the coworking space.
- **Population**: Population of the neighborhood.
- **Percentage of population between 16 and 64 years**: Percentage of the population in this age range.
- **Foreign Population**: Percentage of foreign population.
- **Unemployed**: Number of unemployed people in the neighborhood.
- **Household Average Net Rent**: The average rent for households in the area.
- **Metro**: Whether the coworking space is near a metro station.
- **Closest Bus Stop**: The proximity of the closest bus stop.
- **Mesa Fija**: Fixed desk availability.
- **Mesa Flexible**: Flexible desk availability.
- **Despacho Priado**: Private office availability.
- **Pases/Bonos (días)**: Daily passes or membership prices for the coworking space.

## Analysis and Methodology

### 1. Data Preprocessing
- The data was cleaned and formatted, addressing missing values, categorical features, and outliers.
- Features were normalized and scaled to improve the performance of regression models.

### 2. Price Prediction
The project utilized various regression models to predict coworking space prices based on the following features:

- **Population**
- **Percentage of Population Between 16 and 64 Years**
- **Foreign Population**
- **Unemployment Rate**
- **Household Average Net Rent**
- **Metro Proximity**
- **Desk Types Availability (Fixed, Flexible, Private Offices)**

#### Models Used:
- **Multiple Linear Regression**
- **Polynomial Regression**
- **Ridge Regression**

Each model was evaluated based on the R-squared value and Mean Squared Error (MSE). The Ridge Regression model showed the best fit with a near-perfect R-squared value of 1.0.

### 3. Text Analysis and Word Cloud
The project analyzed reviews from Google Maps to identify the most frequently mentioned words that could help predict customer satisfaction and desired features. A word cloud was generated to visualize the most important keywords based on their frequency and relevance.

### 4. Data Visualization
- **Correlation heatmaps** were generated to visualize the relationship between different features and coworking space prices.
- **Scatter plots** and **line charts** were used to explore relationships between various demographic and price features.

## How to Run the Project

To run this project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/coworking-space-market-analysis.git
cd coworking-space-market-analysis

