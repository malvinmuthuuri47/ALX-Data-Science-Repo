import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.DataFrame({
    'study_hours': [2,3,4,5,6,7,8,9,10,11],
    'exam_scores': [45,50,55,60,65,70,78,82,88,92]
    })

X = df[['study_hours']]
y = df['exam_scores']

model = LinearRegression()
model.fit(X, y)

y_predicted = model.predict(X)

print('Actual values: ', y.values)
print('Predicted values: ', y_predicted.round(2))

r2 = r2_score(y, y_predicted)
print(f'R²: {r2:.4f}')

mae = mean_absolute_error(y, y_predicted)
print(f'MAE: {mae:.4f}')

mse = mean_squared_error(y, y_predicted)
print(f'MSE: {mse:.4f}')

rmse = np.sqrt(mse)
print(f'RMSE: {rmse:.4f}')

print('=' * 40)
print(f'R²: {r2:.4f} -> model explains {r2*100:.2f}% of the data')
print(f'MAE: {mae:.4f} -> average error size')
print(f'MSE: {mse:.4f} -> penalized average error')
print(f'RMSE: {rmse:.4f} -> readable average error')
print('=' * 40)
