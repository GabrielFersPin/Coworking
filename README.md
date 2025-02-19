# 🏢 Coworking Space Analysis & Recommendation System

## 📌 Project Overview
This project analyzes coworking spaces across major cities (Barcelona, Madrid, New York, and Tokyo) to provide insights and build a recommendation system based on pricing, location, and user ratings. The goal is to help professionals and businesses find the best coworking space based on their preferences.

## 🚀 Features
- **Web Scraping**: Extracted coworking space data from Google Maps and other sources.
- **Data Analysis**: Cleaned and processed data, including population, income, transportation, and ratings.
- **Machine Learning**:
  - **Price Prediction**: Implemented **Ridge Regression** and **Random Forest** models to predict day pass prices.
  - **Recommendation System**: Uses a scoring mechanism considering price, rating, and distance from the city center.
- **Interactive Visualization**:
  - **Power BI Dashboards**: Insights on coworking spaces, transportation, and pricing.
  - **Folium Map**: Visual representation of coworking locations with user ratings.
- **Streamlit Web App**: A user-friendly interface to explore coworking options dynamically.

## 📊 Data Sources
- **Google Maps API**: Extracted locations, ratings, and user reviews.
- **Wikipedia & Census Data**: Incorporated neighborhood population and income statistics.
- **Manual Data Collection**: Pricing details for the top-rated coworking spaces.

## 🔍 Data Processing
- Added **distance from city center** as a key metric.
- Computed **weighted rating scores** to account for both rating and user count.
- Feature engineering for predictive modeling.

## 🤖 Machine Learning Models
### **1️⃣ Ridge Regression**
- Best Parameters: `alpha=10.0`
- MAE: **3.58** (Mean Absolute Error)

### **2️⃣ Random Forest Regressor**
- Best Parameters: `max_depth=None`, `n_estimators=100`
- MAE: **15.93**, RMSE: **21.67**

## 🔮 Recommendation System
A **scoring function** ranks coworking spaces by:
1. **Affordability**: (User's max price - Predicted Price)
2. **Quality**: Rating × 10
3. **Convenience**: Distance from the city center (penalized)

## 🌎 Interactive Map Example
Using `folium`, coworking spaces are plotted on a map with circle markers representing rating popularity.

## 🛠 Technologies Used
- **Python**: `pandas`, `scikit-learn`, `folium`, `Streamlit`
- **Machine Learning**: Regression models for price prediction
- **Visualization**: Power BI, Folium, Streamlit

## 💡 How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/GabrielFersPin/Coworking.git
   cd Coworking
   ```
2. Create a Virtual Environment:
    ```bash
    python -m venv new_venv
    ``
3. Activate the Virtual Environment:
    - Windows:
    ```bash
    new_venv\Scripts\activate
    ```
    - MacOS/Linux:
    ```bash
    source new_venv/bin/activate
    ``` 
4. Install Dependecies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 🏆 Acknowledgments
Special thanks to the open-source community and data providers for enabling this analysis.

---

📌 **Author:** Gabriel Fernandes Pinheiro  
🔗 [LinkedIn](https://www.linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

