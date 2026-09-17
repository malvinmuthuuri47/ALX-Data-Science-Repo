import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'study_hours': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    'exam_scores': [45,50,55,60,65,70,78,82,88,92,95,97,99,100,102,105,108,110,112,115]
    })

X = df[['study_hours']]
y = df['exam_scores']

X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
        )

print(f'Total samples: {len(X)}')
print(f'Training samples: {len(X_train)}')
print(f'Test samples: {len(X_test)}')


model = LinearRegression()
model.fit(X_train, y_train)

slope = model.coef_[0]
intercept = model.intercept_

print(f'Slope: {slope:.4f}')
print(f'Intercept: {intercept:.4f}')

y_pred = model.predict(X_test)

print('Actual test scores: ', y_test.values)
print('Predicted test scores: ', y_pred.round(2))

# Performance on training set
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

# Performance on testing set
test_r2 = r2_score(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)

print('=' * 50)
print(f'{"Metric":<10} {"Training":>15} {"Testing":>15}')
print('=' * 50)
print(f'{"R²":<10} {train_r2:>15.4f} {test_r2:>15.4f}')
print(f'{"MAE":<10} {train_mae:>15.4f} {test_mae:>15.4f}')
print(f'{"MSE":<10} {train_mse:>15.4f} {test_mse:>15.4f}')
print(f'{"RMSE":<10} {train_rmse:>15.4f} {test_rmse:>15.4f}')
print('=' * 50)

fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(
        X_train, y_train,
        color='steelblue',
        s=100,
        alpha=0.8,
        edgecolors='white',
        linewidths=1.5,
        label='Training Data',
        zorder=5
        )

ax.scatter(
        X_test, y_test,
        color='coral',
        s=100,
        alpha=0.8,
        edgecolors='white',
        linewidths=1.5,
        label='Testing Data',
        zorder=5
        )

# Regression line - drawn across the full range of x
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
ax.plot(
        x_line,
        model.predict(x_line),
        color='black',
        linewidth=2,
        linestyle='--',
        label=f'Regression Line (R²={test_r2:.4f})'
        )

ax.set_title(
        'Linear Regression - Training vs Testing Data',
        fontsize=14,
        fontweight='bold',
        pad=15
        )
ax.set_xlabel('Study Hours', fontsize=12)
ax.set_ylabel('Exam Score', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
