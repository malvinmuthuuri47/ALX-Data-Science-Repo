import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'house_size':   [50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                     150, 160, 170, 180, 190, 200, 210, 220, 230, 240],
    'num_bedrooms': [1, 1, 2, 2, 2, 3, 3, 3, 4, 4,
                     4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
    'house_age':    [20, 15, 25, 10, 30, 5, 20, 15, 10, 25,
                     5, 30, 15, 20, 10, 5, 25, 30, 15, 10],
    'house_price':  [150000, 180000, 210000, 240000, 265000, 300000,
                     320000, 355000, 375000, 410000, 440000, 465000,
                     495000, 520000, 550000, 580000, 605000, 635000,
                     660000, 690000]
})

X = df.drop('house_price', axis=1)
y = df['house_price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# Predictions on both sets
y_train_pred = tree_model.predict(X_train)
y_test_pred = tree_model.predict(X_test)

# Calculate MSE and R²
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

# Testing metrics
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print('=' * 55)
print(f'{"Metric":<12} {"Training":>18} {"Testing":>18}')
print('=' * 55)
print(f'{"R²":12} {train_r2:>18.4f} {test_r2:>18.4f}')
print(f'{"MSE":<12} {train_mse:>18.2f} {test_mse:>18.2f}')
print(f'{"RMSE":<12} {train_rmse:>18.2f} {test_rmse:>18.2f}')
print('=' * 55)