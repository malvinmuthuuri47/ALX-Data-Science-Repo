"""
This module introduces the concept of decision trees in machine learning
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.DataFrame({
    'LotSize': [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2700],
    'Bedrooms': [2, 2, 3, 3, 4, 4, 5, 5],
    'Age': [30, 25, 20, 18, 15, 10, 8, 5],
    'Price': [120000, 135000, 180000, 210000, 250000, 275000, 320000, 350000]
    })

X = df[['LotSize', 'Bedrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

'''The Decision Trees code snippet'''
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=3, random_state=42)

tree.fit(X_train, y_train)

tree_y_predict = tree.predict(X_test)
print(tree_y_predict)

tree_rmse = np.sqrt(mean_squared_error(y_test, tree_y_predict))
tree_r2 = r2_score(y_test, tree_y_predict)

print("Tree RMSE: ", tree_rmse)
print("Tree R²: ", tree_r2)

plt.figure(figsize=(12,8))

plot_tree(tree, feature_names=X.columns, filled=True, rounded=True)

plt.show()

'''The Random Forest code snippet'''
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)

forest.fit(X_train, y_train)

forest_y_predict = forest.predict(X_test)

forest_rmse = np.sqrt(mean_squared_error(y_test, forest_y_predict))
forest_r2 = r2_score(y_test, forest_y_predict)

print("Forest RMSE: ", forest_rmse)
print("Forest R²: ", forest_r2)