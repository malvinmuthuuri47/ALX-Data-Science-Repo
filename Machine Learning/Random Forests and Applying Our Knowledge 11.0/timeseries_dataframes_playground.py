"""
This module showcases how to filter dataframes content using another dataframe
"""

import pandas as pd
import numpy as np

population_df = pd.DataFrame({
    "2010": [41000000, 35000000, 45000000, 52000000, 21000000, 30000000],
    "2011": [42000000, 36000000, 46300000, 53100000, 21400000, 30600000],
    "2012": [43100000, 37100000, 47700000, 54300000, 21800000, 31200000],
    "2013": [44300000, 38300000, 49200000, 55500000, 22300000, 31800000],
    "2014": [45600000, 39500000, 50800000, 56800000, 22800000, 32500000],
    "2015": [47000000, 40800000, 52500000, 58200000, 23400000, 33200000],
    "2016": [48500000, 42200000, 54300000, 59600000, 24000000, 34000000],
    "2017": [50100000, 43700000, 56200000, 61100000, 24600000, 34800000],
    "2018": [51800000, 45200000, 58200000, 62700000, 25300000, 35600000],
    "2019": [53600000, 46800000, 60300000, 64300000, 26000000, 36500000],
    "2020": [55500000, 48500000, 62500000, 66000000, 26800000, 37400000]
}, index=['KEN', 'UGA', 'TZA', 'ETH', 'RWA', 'ZMB'])

population_df.index.name = 'Country Code'

# print(population_df)

east_african_df = pd.DataFrame({
    'Region': [
        'East Africa',
        'East Africa',
        'East Africa',
        'East Africa'
    ]
}, index=['KEN', 'UGA', 'TZA', 'RWA'])

east_african_df.index.name = 'Country Code'

# print(east_african_df)

def aggregate_population_df_by_region(population_df, region_df):

    selected_countries = region_df.index

    # print(selected_countries)

    filtered_population = population_df.loc[population_df.index.isin(selected_countries)]

    # print(filtered_population)

    results = []

    for year in filtered_population.columns:
        yearly_population = filtered_population[year]
        total_population = yearly_population.sum()

        results.append([
            int(year),
            total_population
        ])

    return np.array(results)

east_african_population = aggregate_population_df_by_region(population_df, east_african_df)
# print(east_african_population)

