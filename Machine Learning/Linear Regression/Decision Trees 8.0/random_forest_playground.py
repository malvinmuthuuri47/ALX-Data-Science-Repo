"""
This module contains code that's meant to showcase how a Random Forest
Regressor can be implemented from sklearn.ensemble to predict a response
variable
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.DataFrame({
    'Temperature': [20,22,24,26,28,30,32,34],
    'Rainfall': [100,110,120,130,140,150,160,170],
    'Fertilizer': [40,42,45,48,50,52,55,58],
    'Pesticide': [10,12,13,14,15,16,18,20],
    'CropYield': [2.1,2.4,2.8,3.2,3.6,3.9,4.3,4.8]
    })

X = df.drop('CropYield', axis=1)
y = df['CropYield']

X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
        )

rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
        )

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE: ", mse)
print("R²: ", r2)

importance = pd.Series(
        rf.feature_importances_,
        index=X.columns
        )
print(importance)
