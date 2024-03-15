import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://ravaneel:qHXdvEKzBoS5AXbc@mongopoc.xvkypvg.mongodb.net/?retryWrites=true"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client.get_database('neel')
records = db.test_data

# Query MongoDB for user data
cursor1 = records.find({}, {'_id': 0, 'country': 1, 'converted': 1, 'device': 1})

# Convert cursor to DataFrame
df = pd.DataFrame(list(cursor1))

# Map converted values to Yes and No
df['converted'] = df['converted'].map({True: 'Yes', False: 'No'})

# Filter based on converted and device
converted_values = st.multiselect("Select converted values:", options=['Yes', 'No'], default=['Yes', 'No'])
device_values = st.multiselect("Select device values:", options=['laptop', 'mobile'], default=['laptop', 'mobile'])

filtered_df = df[(df['converted'].isin(converted_values)) & (df['device'].isin(device_values))]

# Print total number of users
st.write(f"Total number of users: {len(filtered_df)}")

# Group by country and count the number of users
country_counts = filtered_df.groupby('country').size().reset_index(name='count')

# Read world shapefile
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge world shapefile with user data
world = world.merge(country_counts, left_on='name', right_on='country', how='left')

# Set aspect ratio to make the countries closer together
world_aspect_ratio = 2  # Adjust as needed

# Streamlit app
st.title("Number of Users by Country")

# Check if either converted or device values are selected
if converted_values or device_values:
    # Plot map only if selections are made
    if not filtered_df.empty:
        # Plot map
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot countries where converted is Yes in green and converted is No in red
        colors = {'Yes': 'green', 'No': 'red'}
        world.plot(column='count', cmap="RdYlGn", linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
        
        # Set aspect ratio
        ax.set_aspect(world_aspect_ratio)
        
        # Add annotations for number of users
        for idx, row in world.iterrows():
            if not pd.isnull(row['count']):
                ax.text(row.geometry.centroid.x, row.geometry.centroid.y, int(row['count']), fontsize=8, ha='center')
        
        # Display plot in Streamlit
        st.pyplot(fig)
    else:
        st.write("No data available for selected criteria.")
else:
    st.write("Please select at least one option for converted or device to display the map.")

# Query MongoDB for user data
cursor2 = records.find({}, {'_id': 0, 
                            'banalysis': 1, 
                            'bplan': 1, 
                            'competitor': 1, 
                            'gtm': 1, 
                            'brand': 1, 
                            'social': 1, 
                            'content': 1, 
                            'digital': 1, 
                            'pitch': 1, 
                            'investor': 1, 
                            'due': 1, 
                            'elevator': 1, 
                            'pager': 1,
                            'converted': 1})

# Convert cursor to DataFrame
df1 = pd.DataFrame(list(cursor2))

# Prepare data
X = df1[['banalysis', 'bplan', 'competitor', 'gtm', 'brand', 'social', 'content', 'digital', 'pitch', 'investor', 'due', 'elevator', 'pager']]
y = df1['converted']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define neural network architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_scaled, y, epochs=10, batch_size=32)
# Streamlit app
st.title("Conversion Prediction")

# Sliders for feature values
with st.sidebar:
    st.subheader("Feature Sliders")
    banalysis = st.slider("Business Analysis", min_value=0, max_value=5, value=2)
    bplan = st.slider("Business Plan", min_value=0, max_value=5, value=2)
    competitor = st.slider("Competitor Analysis", min_value=0, max_value=5, value=2)
    gtm = st.slider("Go to Market Strategy", min_value=0, max_value=5, value=2)
    brand = st.slider("Brand Strategy", min_value=0, max_value=5, value=2)
    social = st.slider("Social Media Strategy", min_value=0, max_value=5, value=2)
    content = st.slider("Content Marketing Strategy", min_value=0, max_value=5, value=2)
    digital = st.slider("Digital Marketing Strategy", min_value=0, max_value=5, value=2)
    pitch = st.slider("Pitch Deck", min_value=0, max_value=5, value=2)
    investor = st.slider("Investor Brief", min_value=0, max_value=5, value=2)
    due = st.slider("Due Diligence Report", min_value=0, max_value=5, value=2)
    elevator = st.slider("Elevator Pitch", min_value=0, max_value=5, value=2)
    pager = st.slider("One Pager", min_value=0, max_value=5, value=2)

# Predict conversion probability
input_data = np.array([[banalysis, bplan, competitor, gtm, brand, social, content, digital, pitch, investor, due, elevator, pager]])
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data)

# Display predicted probability
st.subheader("Predicted Probability of Conversion")
st.info(f"{prediction[0][0]:.2f}")

# Query MongoDB for user data
cursor3 = records.find({}, {'_id': 0, 'converted': 1})

# Convert cursor3 to DataFrame
df2 = pd.DataFrame(list(cursor3))

# Count converted users
converted_counts = df2['converted'].value_counts()

# Plot bar chart
plt.bar(converted_counts.index.astype(str), converted_counts)
plt.xlabel('Converted')
plt.ylabel('Number of Users')
plt.title('Converted Users')
st.pyplot()
