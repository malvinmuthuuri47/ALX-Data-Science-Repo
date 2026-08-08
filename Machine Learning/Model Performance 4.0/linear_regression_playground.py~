import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Data
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([3,5,4,7,8,9,10,12,11,14,16,15,18,17,20,22,21,24,23,26])

x = x.reshape(-1, 1)

# Split - 80% training, 20% test
X_train, X_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
        )

print(f'Total samples: {len(x)}')
print(f'Training samples: {len(X_train)}')
print(f'Test samples: {len(X_test)}')

# Step 1 - fit the model on the training data only
model = LinearRegression()
model.fit(X_train, y_train)

# Step 2 - evaluate the model on the training data
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = model.score(X_train, y_train)

# Step 3 - evaluate on Test data - data model has never seen
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = model.score(X_test, y_test)

print('=' * 40)
print(f'Training R²: {train_r2:.4f}')
print(f'Training MSE: {train_mse:.4f}')
print('=' * 40)
print(f'Test R²: {test_r2:.4f}')
print(f'Test MSE: {test_mse:.4f}')
print('=' * 40)
