"""
This module is meant to introduce the learner to the concept of concat
and fillna in Pandas Dataframes
"""

import pandas as pd

'''Stacking rows'''
df1 = pd.DataFrame({'name': ['Alice', 'Bob'], 'score': [90, 85]})
df2 = pd.DataFrame({'name': ['Carol', 'Dave'], 'score': [75, 95]})

# print(pd.concat([df1, df2]))

# print(pd.concat([df1, df2], ignore_index=True))

'''Stacking Columns'''
df_intj = pd.DataFrame({'INTJ': [152, 98]}, index=['think', 'logic'])
df_enfp = pd.DataFrame({'ENFP': [88, 12]}, index=['think', 'feel'])

# print(df_intj)
# print(df_enfp)
# print(pd.concat([df_intj, df_enfp], axis=1))

'''fillna'''
combined = pd.concat([df_intj, df_enfp], axis=1)
# print(combined.fillna(0))

'''filling different columns differently'''
df = pd.DataFrame({
    'age': [25, None, 30],
    'city': ['NYC', None, 'LA']
})


df.fillna({'age': 0, 'city': 'UnKnown'}, inplace=True)
# print(df)

'''forward fill'''
df = pd.DataFrame({'temp': [70, None, None, 75]})
df.fillna(method='ffill', inplace=True)
print(df)