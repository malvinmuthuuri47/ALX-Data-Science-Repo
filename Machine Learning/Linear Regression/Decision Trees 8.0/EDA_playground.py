"""
This module is a refresher for EDA, combining EDA, correlation analysis,
variable selection, and implementing Ensemble methods in Machine learning
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.DataFrame({
#     'Temperature': [20,22,24,25,26,27,28,29],
#     'Rainfall': [100,120,140,150,160,170,180,190],
#     'Crop_yield': [2.1,2.3,2.8,3.0,3.1,3.5,3.7,4.0]
#     })


''' Histogram'''
# bins = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]

# plt.figure(figsize=(8,5))

# plt.hist(df['Crop_yield'], bins=bins, edgecolor='white')
# plt.xlabel('Crop Yield')
# plt.ylabel('Frequency')
# plt.title('Distribution of Crop Yield')

# plt.show()

'''Density plot'''
# plt.figure(figsize=(8,5))

# df['Crop_yield'].plot(kind='density')

# plt.xlabel('Crop_yield')
# plt.title('Density Plot')

# plt.show()

'''Density curve + histogram'''
# plt.figure(figsize=(8,5))

# df['Crop_yield'].hist(bins=5, density=True, edgecolor='white')
# df['Crop_yield'].plot(kind='density')

# plt.xlabel('Crop Yield')
# plt.title('Histogram with Density Curve')

# plt.show()

'''Box Plot'''
# plt.figure(figsize=(6,4))

# plt.boxplot(df['Crop_yield'])

# plt.ylabel('Crop Yield')
# plt.title('Box Plot')

# plt.show()

'''Violin Plot'''
# plt.figure(figsize=(6,4))

# plt.violinplot(df['Crop_yield'])

# plt.ylabel('Crop Yield')
# plt.title('Violin Plot')

# plt.show()

df = pd.DataFrame({
    'Temperature': [20,22,24,26,28,30,32,34],
    'Rainfall': [100,110,120,130,140,150,160,170],
    # 'Fertilizer': [40,42,45,47,50,52,55,58],
    'CropYield': [2.1,2.4,2.8,3.0,3.4,3.8,4.1,4.5],
    'Crop': ['Maize', 'Maize', 'Maize', 'Beans', 'Beans', 'Beans', 'Beans', 'Maize']
})

sns.pairplot(df, hue='Crop')

plt.show()