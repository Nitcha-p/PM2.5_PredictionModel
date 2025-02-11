from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, to_date, to_timestamp, hour, dayofweek, sin, cos, when, concat_ws, mean, stddev, \
    date_format, lit, hour, corr, count, row_number
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("PM2.5 Prediction Model").getOrCreate()

# Load data
data_path = "Weather&AirPollution_Collection_Pyspark.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Count the number of unique districts in the "District" column
unique_districts = df.select("District").distinct().count()
print(f"Number of unique districts: {unique_districts}")
# Check Schema
# df.printSchema()

# Convert 'Date (D-M-YYYY)' column to date format and rename it to 'Date'
df = df.withColumn("Date", to_date(col('Date(D_M_YYYY)'), 'd/M/yyyy'))
df = df.drop('Date(D_M_YYYY)')
# df.select('Date').show()

# Extract day of the week with the desired format (0=Monday, 6=Sunday)
df = df.withColumn('Day_of_week', ((dayofweek(col('Date')) + 5) % 7))  # Shift and adjust the week format
# df.select(['Date','Day_of_week']).show()

# Create a weekend indicator (1 for Saturday and Sunday, 0 otherwise)
df = df.withColumn('is_weekend', when(col('Day_of_week').isin(5, 6), 1).otherwise(0))

# Clean 'Time (Hour of a day)' column to extract hours
df = df.withColumn('Hour', hour(col('Time(Hour_of_a_day)')))

# Rename 'Time (Hour_of_a_day)' column
df = df.withColumnRenamed('Time(Hour_of_a_day)', 'Datetime')
# df.printSchema()

# Sort by 'District' and 'Datetime'
df = df.orderBy(['District', 'Datetime'])

# Generate lag features by district
windowSpec = Window.partitionBy('District').orderBy('Datetime')
df = df.withColumn('PM2_5_lag_1h', lag('PM2_5(ug/m3)', 1).over(windowSpec))
df = df.withColumn('PM2_5_lag_6h', lag('PM2_5(ug/m3)', 6).over(windowSpec))
df = df.withColumn('PM2_5_lag_24h', lag('PM2_5(ug/m3)', 24).over(windowSpec))

# Drop rows with missing values
df = df.na.drop()

# Use StringIndexer to encode 'District' - Label encoding
indexer = StringIndexer(inputCol='District', outputCol='District_Label_encoded')
df = indexer.fit(df).transform(df)

correlation_features = [
       'District_Label_encoded', 'Temperature(Celsius)', 'Humidity(Percentage)',
       'WindSpeed(KM/H)', 'Precipitation(Millimeters)',
       'AtmosphericPressure(Millibars)', 'PM10(ug/m3)', 
       'NO2(ug/m3)','SO2(ug/m3)', 'CO(ug/m3)', 'O3(ug/m3)',
       'PM2_5(ug/m3)', 'Day_of_week', 'is_weekend','Hour',
       'PM2_5_lag_1h', 'PM2_5_lag_6h', 'PM2_5_lag_24h']

# Convert to Pandas DataFrame
df_pandas = df.select(correlation_features).toPandas()

# Calculate the correlation matrix
corr_matrix = df_pandas.corr()

# Display the correlation matrix
# print(corr_matrix)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
# plt.show()

# Sort the correlation values with 'PM2.5 (ug/m³)'
target_corr = corr_matrix['PM2_5(ug/m3)'].sort_values(ascending=False)
# print(target_corr)

# Cyclical encoding for 'Hour' (0-23 range)
df = df.withColumn('Hour_sin', sin(2 * 3.14159 * col('Hour') / 24))
df = df.withColumn('Hour_cos', cos(2 * 3.14159 * col('Hour') / 24))

# Cyclical encoding for 'Day_of_week' (0-6 range)
df = df.withColumn('Day_of_week_sin', sin(2 * 3.14159 * col('Day_of_week') / 7))
df = df.withColumn('Day_of_week_cos', cos(2 * 3.14159 * col('Day_of_week') / 7))

print(df.columns)

# Drop rows with missing values
df = df.na.drop()

# Create a sequential number in each District ordered by time
windowSpec = Window.partitionBy("District_Label_encoded").orderBy("Datetime")
df = df.withColumn("row_num", row_number().over(windowSpec))

# คำนวณจำนวนแถวเพื่อแบ่งเป็น Train และ Test (70%-30%)
# คำนวณจำนวนแถวทั้งหมดในแต่ละ District
district_counts = df.groupBy("District_Label_encoded").count()

# เข้าร่วมข้อมูลกับจำนวนแถวในแต่ละ District เพื่อให้ได้ split point
df = df.join(district_counts, "District_Label_encoded")
df = df.withColumn("split_point", (col("count") * 0.7).cast("int"))

# แบ่งข้อมูลเป็น Train และ Test ตาม row_num
train_df = df.filter(col("row_num") <= col("split_point"))
test_df = df.filter(col("row_num") > col("split_point"))

# 7. แปลงข้อมูลเป็น Pandas DataFrame
train_df = train_df.toPandas()
test_df = test_df.toPandas()

# print(train_df.columns)

# ลบคอลัมน์ที่ไม่จำเป็น และจัดเตรียมข้อมูลสำหรับโมเดล
drop_columns = ['PM2_5(ug/m3)', 'Date', 'Datetime', 'District', 'count', 'split_point', 'row_num','AQI(Index/Level)',
                 'Precipitation(Millimeters)', 'Hour','Day_of_week']

X_train = train_df.drop(columns=drop_columns)
y_train = train_df["PM2_5(ug/m3)"]

X_test = test_df.drop(columns=drop_columns)
y_test = test_df["PM2_5(ug/m3)"]

# สร้างและเทรนโมเดล LightGBM
model = LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
# Calculate RMSE by taking the square root of MSE
rmse = mse ** 0.5
# Calculate R score
r2 = r2_score(y_test,y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# ทำนายผลลัพธ์ในชุด Train และ Test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# คำนวณค่า R² สำหรับชุด Train และ Test
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("R² on Train Data:", r2_train)
print("R² on Test Data:", r2_test)

# Visualize Actual vs. Predicted PM2.5
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:100], label='Actual PM2.5', marker='o', linestyle='dashed')
plt.plot(y_pred[:100], label='Predicted PM2.5', marker='x')
plt.title('Actual vs. Predicted PM2.5')
plt.xlabel('Time (Test Data)')
plt.ylabel('PM2.5 (ug/m³)')
plt.legend()
plt.show()

# Stop Spark session
spark.stop()
