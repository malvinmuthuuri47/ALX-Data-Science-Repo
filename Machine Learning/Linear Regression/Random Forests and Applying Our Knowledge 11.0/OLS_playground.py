"""
This module is meant to walk the learner through the process of
fitting a statsmodel OLS Regression model on some data
"""

import pandas as pd
# import statsmodels.api as sm
import statsmodels.formula.api as smf

data = {
    'area': [80, 120, 150, 200, 95, 180, 250, 130, 170, 220, 110, 145, 190, 75, 160],
    'bedrooms': [2, 3, 3, 4, 2, 4, 5, 3, 3, 4, 2, 3, 4, 2, 3],
    'age': [15, 10, 8, 5, 20, 7, 3, 12, 9, 6, 18, 11, 4, 25, 10],
    'distance': [12, 8, 6, 4, 15, 5, 3, 9, 7, 4, 13, 8, 5, 16, 7],
    'parking': [1, 1, 2, 2, 1, 2, 3, 1, 2, 2, 1, 2, 2, 1, 2],
    'price': [85, 125, 155, 215, 92, 190, 275, 135, 175, 230, 105, 150, 205, 78, 165]
}

df = pd.DataFrame(data)

# print(df)

'''Correlation Analysis'''
correlations = df.corr()['price'].abs().sort_values(ascending=False).drop('price')

# print(correlations)

# X = df.drop('price', axis=1)
# y = df['price']

# X = sm.add_constant(X)

# print(X.head())

# model = sm.OLS(y, X).fit()
# print(model.summary())

formula = 'price ~ area + bedrooms + age + distance + parking'

model = smf.ols(formula, data=df).fit()

print(model.summary())