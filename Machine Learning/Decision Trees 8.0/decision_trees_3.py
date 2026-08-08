"""
This module depicts the essence of tuning the max_depth hyperparameter of
a DecisionTreeRegressor and how it affects the model's performance
"""
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
        'Temperature': [22,24,26,28,30,32,34,36],
        'Rainfall': [80,90,110,120,130,150,160,170],
        'Fertilizer': [120,130,150,170,180,200,210,220],
        'Yield': [3.8,4.2,4.8,5.3,5.8,6.4,6.8,7.2]
        }

df = pd.DataFrame(data)

X = df.drop('Yield', axis=1)
y = df['Yield']

X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
        )

depths = []
mse_scores = []
r2_scores = []

for depth in range(1, 11):
    tree = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42
            )

    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    depths.append(depth)
    mse_scores.append(mse)
    r2_scores.append(r2)

results = pd.DataFrame({
    'Tree_depth': depths,
    'MSE': mse_scores,
    'R2': r2_scores
    })

print(results)
