# PM2.5 Prediction Model
This project predicts PM2.5 concentrations by comparing the performance of three machine learning models : LightGBM, XGBoost, and CatBoost. LightGBM was selected as the best model based on MAE, RMSE, and R² metrics. Since PySpark does not natively support LightGBM, the processed Spark DataFrame was converted to a Pandas DataFrame for model training. The project also addresses challenges related to encoding and memory optimization.

# Project Overview
Goal : Build a model to predict PM2.5 levels based on environmental data (e.g., temperature, humidity, wind speed).  
Data : Collected hourly data from APIs, including features like temperature, atmospheric pressure, and pollutant concentrations.  
Models : LightGBM, XGBoost, CatBoost  
Metrics :
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

# Key Features
- Feature Engineering :
  - Extracted features such as Day_of_week, is_weekend, and Hour from datetime data.
  - Created lag features (e.g., PM2.5 lagged by 1, 6, and 24 hours).
  - Applied cyclical encoding to time features and experimented with both one-hot and label encoding for categorical features.
- Model Selection :
  - Models were evaluated based on MAE, RMSE, and R².
  - LightGBM was chosen for its superior performance.
- PySpark Implementation :
  - Due to PySpark's lack of native LightGBM support, the processed Spark DataFrame was converted to Pandas for model training.
  - Label encoding was used instead of one-hot encoding to reduce memory usage, though this impacted accuracy slightly.

# Challenges and Solutions
- Memory Issues : One-hot encoding caused memory constraints in PySpark - Solution : Used label encoding to reduce memory usage.
- Model Support : PySpark does not natively support LightGBM - Solution : Converted Spark DataFrame to Pandas for training.

# Results Summary
- Model Performance (Pandas implementation) :
  - LightGBM : MAE = 10.51, RMSE = 13.73, R² = 0.95
  - XGBoost : MAE = 10.65, RMSE = 13.84, R² = 0.95
  - CatBoost : MAE = 11.66, RMSE = 16.08, R² = 0.93
- Model Performance (Pyspark implementation) :
  - LightGBM : MAE = 14.44, RMSE = 18.53, R² = 0.94
- The Pandas implementation (using one-hot encoding) outperformed the PySpark implementation (using label encoding) due to better handling of categorical features.

# Future Improvements
- Explore alternatives to label encoding, such as distributed one-hot encoding methods.
- Investigate the use of other Spark-compatible models for scalability.

# Acknowledgments
- Data sources : APIs providing environmental data and pollutant concentrations.
- Libraries used : Pandas, PySpark, LightGBM, XGBoost, CatBoost, Scikit-learn.
