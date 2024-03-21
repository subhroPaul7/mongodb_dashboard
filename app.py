import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
from scipy.cluster import hierarchy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from pymongo.mongo_client import MongoClient
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import itertools

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
st.set_option('deprecation.showPyplotGlobalUse', False)


# st.title("Number of Users by Country")
# # Query MongoDB for user data
# cursor1 = records.find({}, {'_id': 0, 'country': 1, 'converted': 1, 'device': 1,})

# # Convert cursor to DataFrame
# df = pd.DataFrame(list(cursor1))

# # Map converted values to Yes and No
# df['converted'] = df['converted'].map({True: 'Yes', False: 'No'})

# # Filter based on converted and device
# converted_values = st.multiselect("Converted:", options=['Yes', 'No'], default=['Yes', 'No'])
# device_values = st.multiselect("Device:", options=['laptop', 'mobile'], default=['laptop', 'mobile'])

# filtered_df = df[(df['converted'].isin(converted_values)) & (df['device'].isin(device_values))]

# # Print total number of users
# st.info(f"Total number of users: {len(filtered_df)}")

# # Group by country and count the number of users
# country_counts = filtered_df.groupby('country').size().reset_index(name='count')
# # Read world shapefile
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# # Merge world shapefile with user data
# world = world.merge(country_counts, left_on='name', right_on='country', how='left')

# # Set aspect ratio to make the countries closer together
# world_aspect_ratio = 2  # Adjust as needed


# # Check if either converted or device values are selected
# if converted_values or device_values:
#     # Plot map only if selections are made
#     if not filtered_df.empty:
#         # Plot map
#         fig, ax = plt.subplots(figsize=(10, 6))
    
#         world.plot(column='count', cmap="RdYlGn", linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
        
#         # Set aspect ratio
#         ax.set_aspect(world_aspect_ratio)
        
#         # Add annotations for number of users
#         for idx, row in world.iterrows():
#             if not pd.isnull(row['count']):
#                 ax.text(row.geometry.centroid.x, row.geometry.centroid.y, int(row['count']), fontsize=8, ha='center')
        
#         # Display plot in Streamlit
#         st.pyplot(fig)
#     else:
#         st.write("No data available for selected criteria.")
# else:
#     st.write("Please select at least one option for converted or device to display the map.")

# # Calculate the number of users in each country
# c_counts = filtered_df['country'].value_counts()
# # Create a pie chart
# fig, ax = plt.subplots()
# ax.pie(c_counts, labels=c_counts.index, autopct='%1.1f%%', startangle=90)
# ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# ax.set_title("Number of Users by Country")

# # Display the pie chart
# st.pyplot(fig)

st.title("Login Page")
# Query MongoDB for data
cursor7 = records.find({}, {'_id': 0, 'si_name_inp': 1, 'si_mail_inp': 1, 'si_pass_inp': 1, 'si_eye_clk': 1, 'lg_duration': 1})

# Convert cursor to DataFrame
df6 = pd.DataFrame(list(cursor7))
# Segregate DataFrame based on lg_duration
df_lg_duration_ge_40 = df6[df6['lg_duration'] >= 40]
df_lg_duration_lt_40 = df6[df6['lg_duration'] < 40]

# Calculate average lg_duration
avg_lg_duration = round(df6['lg_duration'].mean(),2)
# Calculate mean values for specified features where lg_duration >= 40
mean_values_lg_duration_ge_40 = df_lg_duration_ge_40[['si_name_inp', 'si_mail_inp', 'si_pass_inp', 'si_eye_clk']].mean()

# Calculate mean values for specified features where lg_duration < 40
mean_values_lg_duration_lt_40 = df_lg_duration_lt_40[['si_name_inp', 'si_mail_inp', 'si_pass_inp', 'si_eye_clk']].mean()
st.metric(label = "Average Login Session", value = f"{avg_lg_duration} secs")
# Display mean values for specified features where lg_duration >= 40
st.subheader("Mean Click values where login duration exceeds 40 secs")
col7, col8, col9, col10 = st.columns(4)
# Mapping for feature names
feature_mapping = {
    'si_name_inp': 'Name',
    'si_mail_inp': 'Email',
    'si_pass_inp': 'Password',
    'si_eye_clk': 'Eye Icon'
}

# Calculate differences for each feature
delta_si_name = round(mean_values_lg_duration_ge_40['si_name_inp'] - mean_values_lg_duration_lt_40['si_name_inp'], 2)
delta_si_mail = round(mean_values_lg_duration_ge_40['si_mail_inp'] - mean_values_lg_duration_lt_40['si_mail_inp'], 2)
delta_si_pass = round(mean_values_lg_duration_ge_40['si_pass_inp'] - mean_values_lg_duration_lt_40['si_pass_inp'], 2)
delta_si_eye = round(mean_values_lg_duration_ge_40['si_eye_clk'] - mean_values_lg_duration_lt_40['si_eye_clk'], 2)

col7.metric(label="Name", value=round(mean_values_lg_duration_ge_40['si_name_inp'], 2), delta=delta_si_name, delta_color="inverse")
col8.metric(label="Email", value=round(mean_values_lg_duration_ge_40['si_mail_inp'], 2), delta=delta_si_mail, delta_color="inverse")
col9.metric(label="Password", value=round(mean_values_lg_duration_ge_40['si_pass_inp'], 2), delta=delta_si_pass, delta_color="inverse")
col10.metric(label="Eye Icon", value=round(mean_values_lg_duration_ge_40['si_eye_clk'], 2), delta=delta_si_eye, delta_color="inverse")

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

# Replace feature names with provided mapping
feature_mapping = {
    'banalysis': 'Business Analysis',
    'bplan': 'Business Plan',
    'competitor': 'Competitor Analysis',
    'gtm': 'Go to Market Strategy',
    'brand': 'Brand Strategy',
    'social': 'Social Media Strategy',
    'content': 'Content Marketing Strategy',
    'digital': 'Digital Marketing Strategy',
    'pitch': 'Pitch Deck',
    'investor': 'Investor Brief',
    'due': 'Due Diligence Report',
    'elevator': 'Elevator Pitch',
    'pager': 'One Pager'
}

st.title("Strategy Page")

# Query MongoDB to get the sums of each feature
pipeline = [
    {"$group": {"_id": None, "st_idea_clk_sum": {"$sum": "$st_idea_clk"},
                "st_site_clk_sum": {"$sum": "$st_site_clk"},
                "st_docs_clk_sum": {"$sum": "$st_docs_clk"}}}
]
result = list(records.aggregate(pipeline))

sums = result[0]  # Extract the sums from the result
st_idea_clk_sum = sums["st_idea_clk_sum"]
st_site_clk_sum = sums["st_site_clk_sum"]
st_docs_clk_sum = sums["st_docs_clk_sum"]

# Create bar graph
plt.figure(figsize=(10, 6))
bars = plt.bar(["Idea", "Site", "Docs"], [st_idea_clk_sum, st_site_clk_sum, st_docs_clk_sum], color=['blue', 'orange', 'green'])
plt.xlabel('Features')
plt.ylabel('Total clicks')
plt.title('Total Clicks for Each Feature')
# Add labels on the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')

# Display bar graph in Streamlit app
st.pyplot()
# Convert cursor to DataFrame
df1 = pd.DataFrame(list(cursor2))

# Define features and target
X = df1.drop(columns=['converted'])
y = df1['converted']

# Calculate correlation matrix
corr_matrix_st2 = df1.corr()

# Extract correlation values of features with the target variable
corr_with_target = corr_matrix_st2['converted'].drop('converted')


# Generate combinations of 3 features
combinations_3_features = list(combinations(X.columns, 3))

# Train logistic regression model and calculate probabilities for each combination
combination_probabilities = {}
for combination in combinations_3_features:
    X_subset = X[list(combination)]
    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_subset, y)
    # Predict probabilities for class 1 (converted being 1)
    probabilities = model.predict_proba(X_subset)[:, 1]
    # Take the mean probability
    probability = probabilities.mean()
    # Store probability for the combination
    combination_probabilities[combination] = probability

# Get top 5 combinations with highest probability
top_5_combinations = sorted(combination_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]

# Convert feature combinations to strings with proper mapping
mapped_top_5_combinations = [(tuple(feature_mapping.get(feature, feature) for feature in combination), probability) for combination, probability in top_5_combinations]

# Convert feature combinations to strings with proper mapping and formatting
mapped_top_5_combinations_str = [(' & '.join(combination), probability) for combination, probability in mapped_top_5_combinations]

# Create DataFrame for display
top_5_combinations_df = pd.DataFrame(mapped_top_5_combinations_str, columns=["Feature Combination", "Probability of Conversion"])

# Adjust the index to start from 1
top_5_combinations_df.index = np.arange(1, len(top_5_combinations_df) + 1)

# Display top 5 combinations and their probabilities in a table
st.table(top_5_combinations_df)



# Top 3 positively correlated features
top_pos_corr_features = corr_with_target.sort_values(ascending=False).head(3)

# Top 3 negatively correlated features
top_neg_corr_features = corr_with_target.sort_values().head(3)
# Display the top 3 positively correlated features and their effect on converted

# Multiply correlation coefficients by 100 to convert them to percentages
top_pos_corr_features *= 100
top_neg_corr_features *= 100

# Display the top 3 positively correlated features and their effect on converted
st.subheader("Top 3 Positively Correlated Features and Their Effect on Conversion")
col1, col2, col3 = st.columns(3)
for idx, (feature, effect_percentage) in enumerate(zip(top_pos_corr_features.index, top_pos_corr_features.values)):
    if idx == 0:
        col1.metric(label=f":green[{feature_mapping.get(feature, feature)}]", value=f"{effect_percentage:.2f}%")
    elif idx == 1:
        col2.metric(label=f":green[{feature_mapping.get(feature, feature)}]", value=f"{effect_percentage:.2f}%")
    elif idx == 2:
        col3.metric(label=f":green[{feature_mapping.get(feature, feature)}]", value=f"{effect_percentage:.2f}%")

# Display the top 3 negatively correlated features and their effect on converted
st.subheader("Top 3 Negatively Correlated Features and Their Effect on Conversion")
col4, col5, col6 = st.columns(3)
for idx, (feature, effect_percentage) in enumerate(zip(top_neg_corr_features.index, top_neg_corr_features.values)):
    if idx == 0:
        col4.metric(label=f":red[{feature_mapping.get(feature, feature)}]", value=f"{effect_percentage:.2f}%")
    elif idx == 1:
        col5.metric(label=f":red[{feature_mapping.get(feature, feature)}]", value=f"{effect_percentage:.2f}%")
    elif idx == 2:
        col6.metric(label=f":red[{feature_mapping.get(feature, feature)}]", value=f"{effect_percentage:.2f}%")



st.title("GTM Module")
# Query MongoDB for user data
cursor6 = records.find({}, {'_id': 0, 'st_site_gtm_int_clk': 1, 'st_site_gtm_tgm_clk': 1,
                           'st_site_gtm_val_clk': 1, 'st_site_gtm_pri_clk': 1,
                           'st_site_gtm_dis_clk': 1, 'st_site_gtm_mar_clk': 1,
                           'st_site_gtm_sal_clk': 1, 'st_site_gtm_par_clk': 1,
                           'st_site_gtm_kpi_clk': 1, 'st_site_gtm_sav': 1})

# Function to preprocess the target variable
def preprocess_target(value):
    return 1 if value > 0 else 0

# Convert cursor to DataFrame
df5 = pd.DataFrame(list(cursor6))

# Preprocess the target variable
df5['st_site_gtm_sav'] = df5['st_site_gtm_sav'].apply(preprocess_target)

# Extract the feature vector and target variable
X1 = df5[['st_site_gtm_int_clk', 'st_site_gtm_tgm_clk', 'st_site_gtm_val_clk', 
        'st_site_gtm_pri_clk', 'st_site_gtm_dis_clk', 'st_site_gtm_mar_clk', 
        'st_site_gtm_sal_clk', 'st_site_gtm_par_clk', 'st_site_gtm_kpi_clk']]
y1 = df5['st_site_gtm_sav']

# Calculate correlation matrix
corr_matrix_st5 = df5.corr()

# Extract correlation values of features with the target variable
corr_with_target_st5 = corr_matrix_st5['st_site_gtm_sav'].drop('st_site_gtm_sav')

# Define feature mapping
feature_mapping = {
    'st_site_gtm_int_clk': 'Introduction',
    'st_site_gtm_tgm_clk': 'Target Market',
    'st_site_gtm_val_clk': 'Value Proposition',
    'st_site_gtm_pri_clk': 'Pricing Strategy',
    'st_site_gtm_dis_clk': 'Distribution Channels',
    'st_site_gtm_mar_clk': 'Marketing Plan',
    'st_site_gtm_sal_clk': 'Sales Strategy',
    'st_site_gtm_par_clk': 'Partnerships',
    'st_site_gtm_kpi_clk': 'Key Performance Indicators (KPIs)'
}

# Generate combinations of 3 features
combinations_3_features = list(combinations(X1.columns, 3))

# Train logistic regression model and calculate probabilities for each combination
combination_probabilities = {}
for combination in combinations_3_features:
    X_subset = X1[list(combination)]
    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_subset, y1)
    # Predict probabilities for class 1 (st_site_gtm_sav being 1)
    probabilities = model.predict_proba(X_subset)[:, 1]
    # Take the mean probability
    probability = probabilities.mean()
    # Store probability for the combination
    combination_probabilities[combination] = probability

# Get top 5 combinations with highest probability
top_5_combinations = sorted(combination_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]

# Convert feature combinations to strings with proper mapping
mapped_top_5_combinations = [(tuple(feature_mapping.get(feature, feature) for feature in combination), probability) for combination, probability in top_5_combinations]

# Convert feature combinations to strings with proper mapping and formatting
mapped_top_5_combinations_str = [(' & '.join(combination), probability) for combination, probability in mapped_top_5_combinations]

# Create DataFrame for display
top_5_combinations_df = pd.DataFrame(mapped_top_5_combinations_str, columns=["Feature Combination", "Probability of Saving"])

# Adjust the index to start from 1
top_5_combinations_df.index = np.arange(1, len(top_5_combinations_df) + 1)

# Display top 5 combinations and their probabilities in a table
st.table(top_5_combinations_df)

# Get the top 3 positively correlated features
top_pos_corr_features_st5 = corr_with_target_st5.sort_values(ascending=False).head(3)

# Get the top 3 negatively correlated features
top_neg_corr_features_st5 = corr_with_target_st5.sort_values().head(3)

# Multiply correlation coefficients by 100 to convert them to percentages
top_pos_corr_features_st5 *= 100
top_neg_corr_features_st5 *= 100


# Display the top 3 positively correlated features and their effect on saving
st.subheader("Top 3 Positively Correlated Features and Their Effect on Saving")
col1, col2, col3 = st.columns(3)
for idx, (feature, effect_percentage) in enumerate(zip(top_pos_corr_features_st5.index, top_pos_corr_features_st5.values)):
    if idx == 0:
        col1.metric(label=f":green[{feature_mapping.get(feature, feature)}]", value=f"{effect_percentage:.2f}%")
    elif idx == 1:
        col2.metric(label=f":green[{feature_mapping.get(feature, feature)}]", value=f"{effect_percentage:.2f}%")
    elif idx == 2:
        col3.metric(label=f":green[{feature_mapping.get(feature, feature)}]", value=f"{effect_percentage:.2f}%")

# Display the top 3 negatively correlated features and their effect on saving
st.subheader("Top 3 Negatively Correlated Features and Their Effect on Saving")
col4, col5, col6 = st.columns(3)
for idx, (feature, effect_percentage) in enumerate(zip(top_neg_corr_features_st5.index, top_neg_corr_features_st5.values)):
    if idx == 0:
        col4.metric(label=f":red[{feature_mapping.get(feature, feature)}]", value=f"{effect_percentage:.2f}%")
    elif idx == 1:
        col5.metric(label=f":red[{feature_mapping.get(feature, feature)}]", value=f"{effect_percentage:.2f}%")
    elif idx == 2:
        col6.metric(label=f":red[{feature_mapping.get(feature, feature)}]", value=f"{effect_percentage:.2f}%")


# Query MongoDB to fetch required data
cursor8 = records.find({}, {'_id': 0, 'st_site_gtm_bck': 1, 'st_site_gtm_reg': 1, 'converted': 1})

# Convert cursor to DataFrame
df7 = pd.DataFrame(list(cursor8))

# Define feature mapping
feature_mapping = {
    'st_site_gtm_bck': 'Back',
    'st_site_gtm_reg': 'Regenerated',
    'converted': 'Converted'
}

# Calculate correlation matrix
correlation_matrix = df7[['st_site_gtm_bck', 'st_site_gtm_reg', 'converted']].corr() * 100  # Multiply by 100 for percentages

# Map feature names
correlation_matrix = correlation_matrix.rename(index=feature_mapping, columns=feature_mapping)

# Display correlation matrix
st.subheader("Correlation Matrix for Churn:")

# Plot heatmap using Matplotlib
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(correlation_matrix, cmap='viridis', interpolation='nearest')

# Exclude numbers on the diagonal
for i in range(len(correlation_matrix)):
    correlation_matrix.iloc[i, i] = np.nan

plt.colorbar(heatmap, label='Correlation (%)')
plt.xticks(ticks=range(len(correlation_matrix)), labels=correlation_matrix.columns)
plt.yticks(ticks=range(len(correlation_matrix)), labels=correlation_matrix.index)

# Annotate each cell with the correlation value
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix.columns)):
        if not np.isnan(correlation_matrix.iloc[i, j]):  # Exclude NaN values
            plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}%", ha='center', va='center', color='white')

plt.tight_layout()
st.pyplot(plt)
# # Standardize features
# scaler1 = StandardScaler()
# X_scaled1 = scaler1.fit_transform(X1)

# model1 = Sequential([
#     Dense(64, activation='relu', input_shape=(9,)),
#     Dense(128, activation='relu'),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model on the whole dataset
# model1.fit(X_scaled1, y1, epochs=10, batch_size=32)

# # Streamlit app
# st.title("Conversion Prediction based on Strategy page")

# # Sliders for feature values
# with st.sidebar:
#     st.subheader("GTM modules")
#     introduction = st.slider("Introduction", min_value=0, max_value=5, value=2)
#     target_market = st.slider("Target Market", min_value=0, max_value=5, value=2)
#     value_proposition = st.slider("Value Proposition", min_value=0, max_value=5, value=2)
#     pricing_strategy = st.slider("Pricing Strategy", min_value=0, max_value=5, value=2)
#     distribution_channels = st.slider("Distribution Channels", min_value=0, max_value=5, value=2)
#     marketing_plan = st.slider("Marketing Plan", min_value=0, max_value=5, value=2)
#     sales_strategy = st.slider("Sales Strategy", min_value=0, max_value=5, value=2)
#     partnerships = st.slider("Partnerships", min_value=0, max_value=5, value=2)
#     kpis = st.slider("Key Performance Indicators (KPIs)", min_value=0, max_value=5, value=2)


# # Predict conversion probability
# input_data1 = np.array([[introduction, target_market, value_proposition, pricing_strategy, distribution_channels, marketing_plan, sales_strategy, partnerships, kpis]])
# input_data_scaled1 = scaler1.transform(input_data1)
# prediction1 = model1.predict(input_data1)

# # Display predicted probability
# st.subheader("Predicted Probability of Saving GTM strategy")
# st.info(f"{prediction1[0][0]:.2f}")

# corr_matrix = df1.corr()
# # Generate a hierarchical clustering dendrogram
# plt.figure(figsize=(12, 8))
# dendrogram = hierarchy.dendrogram(hierarchy.linkage(corr_matrix, method='ward'), labels=corr_matrix.columns, leaf_rotation=90)

# # Display the plot
# st.pyplot(plt)

# # Extract correlation values of features with the target variable
# corr_with_target = corr_matrix['converted'].drop('converted')

# # Top 3 features positively correlated with the target variable
# top_pos_corr_features = corr_with_target.sort_values(ascending=False).head(3)

# # Top 3 features negatively correlated with the target variable
# top_neg_corr_features = corr_with_target.sort_values().head(3)


# # Display the results in Streamlit"
# st.subheader("Top Features Correlated with Conversion")
# st.metric(label=top_pos_corr_features.index[0], value=top_pos_corr_features.iloc[0])
# st.metric(label=top_pos_corr_features.index[1], value=top_pos_corr_features.iloc[1])
# st.metric(label=top_pos_corr_features.index[2], value=top_pos_corr_features.iloc[2])

# st.subheader("Top Features Negatively Correlated with Conversion")
# st.metric(label=top_neg_corr_features.index[0], value=top_neg_corr_features.iloc[0])
# st.metric(label=top_neg_corr_features.index[1], value=top_neg_corr_features.iloc[1])
# st.metric(label=top_neg_corr_features.index[2], value=top_neg_corr_features.iloc[2])

# # Prepare data
# X = df1[['banalysis', 'bplan', 'competitor', 'gtm', 'brand', 'social', 'content', 'digital', 'pitch', 'investor', 'due', 'elevator', 'pager']]
# y = df1['converted']

# # Standardize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Define neural network architecture
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(13,)),
#     Dense(128, activation='relu'),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_scaled, y, epochs=10, batch_size=32)
# Streamlit app
# st.title("Conversion Prediction based on Strategy page")

# Sliders for feature values
# with st.sidebar:
#     st.subheader("Feature Sliders")
#     banalysis = st.slider("Business Analysis", min_value=0, max_value=5, value=2)
#     bplan = st.slider("Business Plan", min_value=0, max_value=5, value=2)
#     competitor = st.slider("Competitor Analysis", min_value=0, max_value=5, value=2)
#     gtm = st.slider("Go to Market Strategy", min_value=0, max_value=5, value=2)
#     brand = st.slider("Brand Strategy", min_value=0, max_value=5, value=2)
#     social = st.slider("Social Media Strategy", min_value=0, max_value=5, value=2)
#     content = st.slider("Content Marketing Strategy", min_value=0, max_value=5, value=2)
#     digital = st.slider("Digital Marketing Strategy", min_value=0, max_value=5, value=2)
#     pitch = st.slider("Pitch Deck", min_value=0, max_value=5, value=2)
#     investor = st.slider("Investor Brief", min_value=0, max_value=5, value=2)
#     due = st.slider("Due Diligence Report", min_value=0, max_value=5, value=2)
#     elevator = st.slider("Elevator Pitch", min_value=0, max_value=5, value=2)
#     pager = st.slider("One Pager", min_value=0, max_value=5, value=2)

# # Predict conversion probability
# input_data = np.array([[banalysis, bplan, competitor, gtm, brand, social, content, digital, pitch, investor, due, elevator, pager]])
# input_data_scaled = scaler.transform(input_data)
# prediction = model.predict(input_data)

# # Display predicted probability
# st.subheader("Predicted Probability of Conversion")
# st.info(f"{prediction[0][0]:.2f}")

# Plot the graph
st.title("User Activity by Hour of the Day")
# Query MongoDB for user data
cursor3 = records.find({}, {'_id': 0, 'lg_time': 1, 'country': 1, 'device': 1})

# Convert cursor to DataFrame
df2 = pd.DataFrame(list(cursor3))

# Convert lg_time to datetime
df2['lg_time'] = pd.to_datetime(df2['lg_time'])

# Extract the hour part from the datetime
df2['hour'] = df2['lg_time'].dt.hour

# Filter DataFrame based on location (country) and device
selected_countries = st.multiselect('Select countries:', options=["India", "United States of America", "United Kingdom", "Canada", "Australia", "France", "Germany", "China", "Japan", "Brazil"], default=["India", "United States of America", "United Kingdom", "Canada", "Australia", "France", "Germany", "China", "Japan", "Brazil"])
selected_devices = st.multiselect('Select devices:', options=["laptop", "mobile"], default=["laptop", "mobile"])

filtered_df2 = df2[(df2['country'].isin(selected_countries)) & (df2['device'].isin(selected_devices))]

# If either filter is not selected, display a blank plot
if not selected_countries or not selected_devices:
    st.write("Please select at least one option for both countries and devices to display the plot.")
else:
    # Group data by hour and count the number of users for each country
    hourly_counts = filtered_df2.groupby(['hour', 'country']).size().unstack(fill_value=0)

    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10.colors[:len(selected_countries)]  # Get colors from default colormap

    # Stack the bars for each country
    bottom = None
    for country in selected_countries:
        counts = hourly_counts[country]
        plt.bar(counts.index, counts.values, bottom=bottom, label=country, color=colors[selected_countries.index(country)])
        if bottom is None:
            bottom = counts.values
        else:
            bottom += counts.values

    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Users')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Display the plot
    st.pyplot()

# Define column names and their corresponding filter labels
columns = {
    "si_google_new": "Signed in with Google",
    "si_acc_new": "Created account",
    "si_name_inp": "Name input",
    "si_mail_inp": "Mail input",
    "si_pass_inp": "Password input",
    "si_eye_clk": "Eye click",
    "rem_me": "Remember me"
}

# Query MongoDB for user data
cursor4 = records.find({}, {'_id': 0, 'si_google_new':1, 'si_acc_new':1, 'si_name_inp':1, 'si_mail_inp':1, 'si_pass_inp':1, 'si_eye_clk':1, 'rem_me':1})

st.title("Signup Activity")

# Convert cursor to DataFrame
df3 = pd.DataFrame(list(cursor4))
# Map converted values to Yes and No
df3['si_google_new'] = df3['si_google_new'].map({True: 'Yes', False: 'No'})
df3['si_acc_new'] = df3['si_acc_new'].map({True: 'Yes', False: 'No'})
df3['rem_me'] = df3['rem_me'].map({True: 'Yes', False: 'No'})
# Filter DataFrame based on location (country) and device
google = st.multiselect('Continued with Google:', options=["Yes", "No"], default = ["Yes"])
signin = st.multiselect('Created Account:', options=["Yes", "No"], default = ["No"])
rem_me = st.multiselect('Remember me', options=["Yes", "No"], default=["Yes"])

filtered_df3 = df3[(df3['si_acc_new'].isin(signin)) & (df3['si_google_new'].isin(google)) & (df3['rem_me'].isin(rem_me))]

# Calculate the sum of specific columns and the count of rows for the remaining columns
sum_columns = filtered_df3[['si_name_inp', 'si_mail_inp', 'si_pass_inp', 'si_eye_clk']].sum()

row_counts = filtered_df3[['si_google_new', 'si_acc_new', 'rem_me']].shape[0]
st.info(f"Number of users: {row_counts} ")
# Calculate the average based on row_counts
avg = sum_columns / row_counts
# Custom row names
row_names = ["Name", "Email", "Password", "Eye-icon"]
# Plot the sums and row counts in a bar chart
plt.figure(figsize=(10, 6))

# Plot the sums of specific columns
sum_columns.plot(kind='bar', color='skyblue')

plt.title("Clicks per Button/Input field")
plt.ylabel("Count")
plt.legend()
plt.xticks([0, 1, 2, 3], ["Name", "Email", "Password", "Eye Icon"])
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
st.pyplot()

st.subheader("Average Clicks for Features")
col11, col12, col13, col14 = st.columns(4)
col11.metric(label=row_names[0], value=f"{avg[0]:.2f}")
col12.metric(label=row_names[1], value=f"{avg[1]:.2f}")
col13.metric(label=row_names[2], value=f"{avg[2]:.2f}")
col14.metric(label=row_names[3], value=f"{avg[3]:.2f}")

st.title("PDF vs Slides Downloads")
# Query MongoDB for user data
cursor5 = records.find({}, {'_id': 0, 'st_site_gtm_pdf': 1, 'st_site_gtm_ppt': 1, 'converted': 1})

# Convert cursor to DataFrame
df4 = pd.DataFrame(list(cursor5))

# Filter based on converted field
converted_values = st.multiselect("Converted:", options=[True, False], default=[True, False])
filtered_df4 = df4[df4['converted'].isin(converted_values)]

# Calculate sum of 'st_site_gtm_pdf' and 'st_site_gtm_ppt'
sum_pdf = filtered_df4['st_site_gtm_pdf'].sum()
sum_ppt = filtered_df4['st_site_gtm_ppt'].sum()

# Create pie chart
plt.figure(figsize=(2,2))
plt.pie([sum_pdf, sum_ppt], labels=['PDF', 'Slides'], autopct='%1.1f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

# Display pie chart
st.pyplot()

