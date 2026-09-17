"""
This module serves as an example to try and implement time-series analysis
for a Pandas Dataframe.
"""

import pandas as pd
import numpy as np

import pandas as pd

data = {
    "1960": [50000, 45000, 60000, 72000, 81000],
    "1961": [51150, 46050, 61450, 73600, 82700],
    "1962": [52320, 47130, 62980, 75250, 84450],
    "1963": [53510, 48240, 64560, 76950, 86250],
    "1964": [54740, 49380, 66190, 78700, 88100],
    "1965": [56010, 50550, 67870, 80500, 90000],
    "1966": [57320, 51760, 69610, 82350, 91950],
    "1967": [58670, 53010, 71420, 84250, 93950],
    "1968": [60060, 54300, 73310, 86200, 96000],
    "1969": [61490, 55630, 75280, 88200, 98100],
    "1970": [62960, 57000, 77330, 90250, 100250],
    "1971": [64470, 58410, 79470, 92350, 102450],
    "1972": [66020, 59870, 81700, 94500, 104700],
    "1973": [67620, 61380, 84030, 96700, 107000],
    "1974": [69270, 62940, 86470, 98950, 109350],
    "1975": [70970, 64560, 89020, 101250, 111750],
    "1976": [72720, 66230, 91690, 103600, 114200],
    "1977": [74520, 67960, 94480, 106000, 116700],
    "1978": [76380, 69750, 97400, 108450, 119250],
    "1979": [78300, 71600, 100450, 110950, 121850],
    "1980": [80280, 73510, 103650, 113500, 124500],
    "1981": [82320, 75480, 107000, 116100, 127200],
    "1982": [84420, 77510, 110500, 118750, 129950],
    "1983": [86580, 79600, 114160, 121450, 132750],
    "1984": [88800, 81760, 117980, 124200, 135600],
    "1985": [91080, 83980, 121970, 127000, 138500],
    "1986": [93420, 86270, 126130, 129850, 141450],
    "1987": [95820, 88620, 130470, 132750, 144450],
    "1988": [98280, 91040, 134990, 135700, 147500],
    "1989": [100800, 93530, 139700, 138700, 150600],
    "1990": [103380, 96090, 144600, 141750, 153750]
}

countries = ["KEN", "UGA", "TZA", "RWA", "ETH"]

df = pd.DataFrame(data, index=countries)
df.index.name = 'Country'

def get_population_growth_rate_by_country_year(df, country_code):
    country_series = df.loc[country_code]

    results = []
    results2 = []

    for i in range(1, len(country_series)):
        prev_pop = country_series.iloc[i - 1]
        curr_pop = country_series.iloc[i]

        year = country_series.index[i]

        growth_rate = (curr_pop - prev_pop) / prev_pop

        results.append([
            float(year),
            round(growth_rate, 5)
        ])

        results2.append([
            year,
            round(growth_rate, 5)
        ])

    # return np.array(results2)
    return np.array(results)

growth = get_population_growth_rate_by_country_year(df, 'UGA')
# print(growth)

def even_odd_train_test_split(data):
    X = data[:, 0]
    y = data[:, 1]

    # Split masks
    even_mask = X % 2 == 0

    X_train = X[even_mask]
    X_test = X[~even_mask]
    y_train = y[even_mask]
    y_test = y[~even_mask]

    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = even_odd_train_test_split(growth)

print(y_train)