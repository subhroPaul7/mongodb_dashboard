# Streamlit MongoDB Analytics Dashboard

This Streamlit application provides an interactive dashboard for analyzing user data stored in a MongoDB database. The dashboard includes visualizations and insights on user activities, conversion rates, and more.

## Features

### 1. Number of Users by Country
- Users can filter data based on conversion status and device type.
- Displays a choropleth map showing the distribution of users across different countries.
- Also includes a pie chart illustrating the percentage of users in each country.

### 2. Conversion Prediction based on Strategy Page
- Predicts the conversion probability based on user input regarding various strategy-related factors.
- Users can adjust sliders to input values for different strategies.
- Displays the predicted probability of conversion.

### 3. User Activity by Hour of the Day
- Allows users to select specific countries and devices for analysis.
- Shows a bar chart representing the number of users active during each hour of the day for the selected countries and devices.

### 4. Signup Activity
- Enables filtering of signup activity based on Google sign-in, account creation, and 'Remember me' feature.
- Presents a bar chart showing the counts of clicks on different buttons and input fields (e.g., Name, Email, Password).
- Additionally, displays the average number of clicks for each input field.

### 5. PDF vs Slides Downloads
- Allows filtering based on conversion status.
- Presents a pie chart comparing the total number of downloads for PDF and Slides.

## Requirements

- Python 3.7 or later
- Streamlit
- pandas
- geopandas
- matplotlib
- pymongo
- numpy
- TensorFlow
- scikit-learn

## How to Run

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Ensure MongoDB is running and accessible.
4. Replace the MongoDB URI in the `uri` variable with your connection string.
5. Run the Streamlit app using `streamlit run app.py`.
6. Access the Streamlit app in your web browser at the provided URL.
