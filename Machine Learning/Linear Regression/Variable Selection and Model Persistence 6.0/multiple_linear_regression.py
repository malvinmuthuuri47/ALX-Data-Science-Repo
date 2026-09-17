"""
This module contains the code that's meant to depict how multiple linear
regression is implemented for a linear regression model
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = {
        'Mileage': [30000, 50000, 20000, 80000, 40000, 60000, 25000, 7000],
        'Age': [2, 4, 1, 6, 3, 5, 2, 5],
        'Fuel': [
            'Petrol', 'Diesel', 'Petrol', 'Diesel',
            'Petrol', 'Diesel', 'Petrol', 'Diesel'
            ],
        'Transmission': [
            'Manual', 'Automatic', 'Automatic', 'Manual',
            'Manual', 'Automatic', 'Automatic', 'Manual'
            ],
        'Price': [
            22000, 19000, 27000, 15000,
            21000, 17000, 25000, 16000
            ]
        }

df = pd.DataFrame(data)
# print(df)

# dummy encode the categorical variables
df_encoded = pd.get_dummies(
        df,
        columns=['Fuel', 'Transmission'],
        drop_first=True
        )

# print(df_encoded)

# Separate X and y
X = df_encoded.drop(columns='Price')
y = df_encoded['Price']

# split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42
        )

# Create the model
model = LinearRegression()

# train the model
model.fit(X_train, y_train)

# Inspect the learned coefficients
print('Intercept: ', model.intercept_)

# for feature, coef in zip(X.columns, model.coef_):
#     print(feature, ":", coef)

# Make predictions
predictions = model.predict(X_test)
print(predictions)

# Evaluate the model
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print('R² Score: ' , r2)
print('MAE Val: ', mae)
print('MSE Val: ', mse)

results = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': predictions
    })

print(results)
