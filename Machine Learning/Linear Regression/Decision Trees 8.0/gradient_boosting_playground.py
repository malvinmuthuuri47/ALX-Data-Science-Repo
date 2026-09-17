"""
This module provides the learner with an example meant to walk them
through the process of understanding the concept of Gradient Boosting
in ensemble machine learning models
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.DataFrame({
    'Temperature': [20,22,24,26,28,30,32,34],
    'Rainfall': [100,110,120,130,140,150,160,170],
    'Fertilizer': [40,42,45,48,50,52,55,58],
    'Pesticide': [10,12,13,14,15,16,18,20],
    'CropYield': [2.1,2.4,2.8,3.2,3.6,3.9,4.3,4.8]
    })

'''Separate the features and target'''
X = df.drop('CropYield', axis=1)
y = df['CropYield']

'''Split into training and testing data'''
X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
        )

'''Create the Gradient Boosting model'''
gbr = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
        )

'''Train the model'''
gbr.fit(X_train, y_train)

'''Make the predictions'''
y_pred = gbr.predict(X_test)

'''Evaluate the model'''
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE: ", mse)
print("R²: ", r2)

'''Feature Importance'''
importance = pd.Series(
        gbr.feature_importances_,
        index=X.columns
        )

importance = importance.sort_values(ascending=False)
print(importance)
