"""
This module showcases the difference between the sklearn Linear Regression
model and statsmodels.api Ordinary Least Squares Regression class
"""

'''Statsmodels'''
import statsmodels.api as sm
X = sm.add_constant([[1],[2],[3],[4]])
y = [2,4,6,8]

model = sm.OLS(y, X).fit()
print(model.summary())


'''scikit-learn'''
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1],[2],[3],[4]])
y = [2,4,6,8]

model = LinearRegression().fit(X, y)
print(model.coef_, model.intercept_)
