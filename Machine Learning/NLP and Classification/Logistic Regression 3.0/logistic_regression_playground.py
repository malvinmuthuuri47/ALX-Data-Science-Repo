"""
This module is intended to help the learner practice and dissect the
concept of logistic regression
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# np.random.seed(42)

# no. of students
# n = 100

# hours studied: between 1 and 10
# hours_studied = np.random.uniform(1, 10, n)
# print(f'Hours studied: {hours_studied}')
# print(hours_studied.shape)
# print('\n')

# probability of passing increases as study hours increase
# probability_pass = 1 / (1 + np.exp(-(hours_studied - 5)))
# print(f'Probability array: {probability_pass}')
# print(probability_pass.shape)
# print('\n')

# Generate the actual outcome
# passed = np.random.binomial(1, probability_pass)

# df = pd.DataFrame({
#     "hours_studied": hours_studied,
#     "passed": passed
# })

# print(df.head())

# X = df.drop('passed', axis=1)
# y = df['passed']

# model = LogisticRegression()

# model.fit(X, y)

# print(model.coef_)

# probabilities = model.predict_proba(X)
# predictions = model.predict(X)
# print(f'Probabilities: {probabilities.shape}')
# print(f'Predictions: {predictions.shape}')

# inspecting the model
# print(f'Model coefficient: {model.coef_}')
# print(f'Model Intercept: {model.intercept_}')

# predictions for specific students
# new_students = pd.DataFrame({
#     "hours_studied": [2, 4, 5, 6, 8, 10]
# })

# print(model.predict_proba(new_students))

df = pd.DataFrame({
    "age": [
        22, 25, 28, 31, 34, 37, 40, 43, 46, 49,
        52, 55, 58, 61, 64, 67, 29, 35, 42, 51
    ],
    "bmi": [
        21.5, 24.1, 27.8, 30.2, 25.6,
        32.1, 28.4, 35.2, 26.8, 31.5,
        29.7, 33.1, 27.2, 36.4, 30.8,
        34.2, 23.9, 28.7, 32.8, 26.1
    ],
    "smoker": [
        "no", "no", "yes", "no", "no",
        "yes", "no", "yes", "no", "yes",
        "no", "yes", "no", "yes", "no",
        "yes", "no", "no", "yes", "no"
    ],
    "claim_amount": [
        1200, 1500, 4200, 1800, 2100,
        6500, 2400, 8200, 2700, 7100,
        3200, 9000, 3500, 10500, 4100,
        9800, 1900, 2600, 6200, 3000
    ]
})

# sns.displot(
#     data=df,
#     x="claim_amount",
#     hue="smoker",
#     kind="kde",
#     # kde=True
# )

sns.countplot(data=df, x="smoker")

sns.displot(
    data=df,
    x="claim_amount",
    bins=20
)

# sns.displot(
#     data=df,
#     x="age"
# )

# sns.scatterplot(
#     data=df,
#     x="age",
#     y="claim_amount"
# )

plt.show()