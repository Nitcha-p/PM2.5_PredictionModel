import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the CSV file 
file_path = 'Weather&AirPollution_Collection.csv'
df = pd.read_csv(file_path)

# Count the number of unique districts in the "District" column
unique_districts = df['District'].nunique()

# Convert Date to datetime format and Rename column
df['Date (D-M-YYYY)'] = pd.to_datetime(df['Date (D-M-YYYY)'], format='%d/%m/%Y')
df.rename(columns={'Date (D-M-YYYY)':'Date'},inplace=True)

# Extract day of the week and weekend indicator
df['Day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['is_weekend'] = df['Day_of_week'].isin([5, 6]).astype(int)  # 1 if weekend, 0 otherwise

# Clean Time column to Extract hours
df['Time (Hour of a day)'] = df['Time (Hour of a day)'].astype(str)
df['Hour'] = df['Time (Hour of a day)'].str.replace(':00', '').str.strip().astype(int)
df.drop('Time (Hour of a day)', axis=1, inplace=True)

# Combine Date and Hour into a datetime column
df['Datetime'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h')

# Sort by District and Datetime
df = df.sort_values(by=['District', 'Datetime'])

# Create Time-based and Lag Features
df['PM2.5_lag_1h'] = df.groupby('District')['PM2.5 (ug/m3)'].shift(1)
df['PM2.5_lag_6h'] = df.groupby('District')['PM2.5 (ug/m3)'].shift(6)
df['PM2.5_lag_24h'] = df.groupby('District')['PM2.5 (ug/m3)'].shift(24)
# Drop rows with missing lag values
df = df.dropna()

# Calculate the Correlation Matrix
# Temporary Label Encoding
le = LabelEncoder()
df['District_Label_encoded'] = le.fit_transform(df['District'])

correlation_features = [
       'District_Label_encoded', 'Temperature (Celsius)', 'Humidity (%)',
       'Wind Speed (KM/H)', 'Precipitation (Millimeters)',
       'Atmospheric Pressure (Millibars)', 'PM10 (ug/m3)', 
       'NO2 (ug/m3)','SO2 (ug/m3)', 'CO (ug/m3)', 'O3 (ug/m3)',
       'PM2.5 (ug/m3)', 'Day_of_week', 'is_weekend','Hour',
       'PM2.5_lag_1h', 'PM2.5_lag_6h', 'PM2.5_lag_24h']

corr_matrix = df[correlation_features].corr()

# Show the result as Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Select features highly correlated with PM2.5
target_corr = corr_matrix['PM2.5 (ug/m3)'].sort_values(ascending=False)
print(target_corr)


# Cyclical Encoding for Hour
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

# Cyclical Encoding for Day of week
df['Day_of_week_sin'] = np.sin(2 * np.pi * df['Day_of_week'] / 7)
df['Day_of_week_cos'] = np.cos(2 * np.pi * df['Day_of_week'] / 7)

# Apply one-hot encoding to the 'District' column
district_Hot_encoded = pd.get_dummies(df['District'], prefix='District')

# Concatenate the one-hot encoded columns with the original dataframe
df = pd.concat([df, district_Hot_encoded], axis=1)
# print(df.columns)
# Ensure the dataframe is sorted by 'District' and 'Datetime'
df = df.sort_values(by=['District', 'Datetime'])
print(df.columns)

# Initialize empty dataframes to store train and test sets
train_data = pd.DataFrame()
test_data = pd.DataFrame()

# Split data within each district
for district in df['District_Label_encoded'].unique():
    district_data = df[df['District_Label_encoded'] == district]  # Filter data for this district

    # Determine the split point
    train_size = int(0.7 * len(district_data))

    # Split data chronologically
    train_data = pd.concat([train_data, district_data[:train_size]])
    test_data = pd.concat([test_data, district_data[train_size:]])

# Separate features and target for train and test sets
X_train = train_data.drop(columns=['PM2.5 (ug/m3)','Date','Datetime','District','AQI (Index/Level)',
                      'District_Label_encoded','Precipitation (Millimeters)','Hour',
                      'Day_of_week'])  # Drop the target and timestamp
y_train = train_data['PM2.5 (ug/m3)']

X_test = test_data.drop(columns=['PM2.5 (ug/m3)','Date','Datetime','District','AQI (Index/Level)',
                      'District_Label_encoded','Precipitation (Millimeters)','Hour',
                      'Day_of_week'])
y_test = test_data['PM2.5 (ug/m3)']

# Initialize and train the LightGBM model
model = LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Save the model to a file
# with open('LightGBM_PM2.5.pkl', 'wb') as file:
#     pickle.dump(model, file)

y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
# calculate RMSE by taking the square root of MSE
rmse = mse ** 0.5
# calculate R score
r2 = r2_score(y_test,y_pred)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Coefficient of Determination (R2 Score): {r2:.2f}')

# Visualize Actual vs. Predicted PM2.5
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:100], label='Actual PM2.5', marker='o', linestyle='dashed')
plt.plot(y_pred[:100], label='Predicted PM2.5', marker='x')
plt.title('Actual vs. Predicted PM2.5')
plt.xlabel('Time (Test Data)')
plt.ylabel('PM2.5 (ug/m³)')
plt.legend()
plt.show()
