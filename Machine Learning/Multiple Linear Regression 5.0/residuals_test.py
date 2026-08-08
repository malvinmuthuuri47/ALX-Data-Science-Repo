"""
This module depicts an example that's meant to showcase how to visualise
the relationship between predictor variables and model residuals to assess
the independence of the residuals using scatter plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import math

data = {
        'House_size': [80, 100, 120, 150, 180, 200, 220],
        'Bedrooms': [2, 3, 3, 4, 5, 5, 6],
        'Age': [25, 18, 12, 8, 4, 2, 1],
        'Distance': [15, 12, 10, 8, 5, 4, 3],
        'Price': [180, 220, 260, 320, 410, 450, 500]
        }

df = pd.DataFrame(data)

X = df[['House_size', 'Bedrooms', 'Age', 'Distance']]
y = df['Price']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

# Obtain diagnostics
fitted_values = model.fittedvalues
residuals = model.resid

influence = model.get_influence()
cooks_distance = influence.cooks_distance[0]

# Rule of thumb threshold
threshold = 4 / len(df)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(fitted_values, residuals)
plt.axhline(0, color='red', linestyle='--')

for i in range(len(df)):
    if cooks_distance[i] > threshold:
        plt.annotate(
                i,
                (fitted_values[i], residuals[i]),
                color='red'
                )

plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values\n(Cook\'s Distance Outliers Labelled')


# Obtain normalised (internally studentized) residuals
normalized_residuals = model.get_influence().resid_studentized_internal

# Create Q-Q plot
plt.figure(figsize=(8,6))
stats.probplot(
        normalized_residuals,
        dist='norm',
        plot=plt
        )


# Perform the Breusch-Pagan test
lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(
        model.resid, model.model.exog
        )

print(f'LM Statistic : {lm_stat:.4f}')
print(f'LM p-value : {lm_pvalue:.4f}')
print(f'F Statistic : {f_stat:.4f}')
print(f'F p-value: {f_pvalue:.4f}')

if lm_pvalue > 0.05:
    print('\nConclusion:')
    print('The model satisfies the homoskedasticity assumption.')
else:
    print('\nConclusion:')
    print('The model violates the homoscedasticity assumption.')

residuals = model.resid

predictors = ['House_size', 'Bedrooms', 'Age', 'Distance']

num_plots = len(predictors)

cols = math.ceil(math.sqrt(num_plots))
rows = math.ceil(num_plots / cols)

fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

axs = axs.flatten()

for i, predictor in enumerate(predictors):
    axs[i].scatter(df[predictor], residuals)

    # Horizontal reference line at zero
    axs[i].axhline(0, color='red', linestyle='--')

    axs[i].set_title(f'{predictor} vs Residuals')
    axs[i].set_xlabel(predictor)
    axs[i].set_ylabel('Residual')

plt.tight_layout()
plt.show()
