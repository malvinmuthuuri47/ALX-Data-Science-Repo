"""
This module is meant to help the learner understand the F statistic and
Prob(F-statistic) in an smf.OLS() model's summary
"""

import pandas as pd
import statsmodels.formula.api as smf

df = pd.DataFrame({
    'house_size': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240],
    'num_bedrooms': [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
    'house_age': [20, 15, 25, 10, 30, 5, 20, 15, 10, 25, 5, 30, 15, 20, 10, 5, 25, 30, 15, 10],
    'house_price': [150000, 180000, 210000, 240000, 265000, 300000, 320000, 355000, 375000, 410000, 440000, 465000, 495000, 520000, 550000, 580000, 605000, 635000, 660000, 690000]
    })

# print(df)

model = smf.ols(
    formula='house_price ~ house_size + num_bedrooms + house_age',
    data=df
).fit()

# print(model.summary())
print(f'F-statistic: {model.fvalue:.4f}')
print(f'Prob (F-statistic): {model.f_pvalue:.8f}')