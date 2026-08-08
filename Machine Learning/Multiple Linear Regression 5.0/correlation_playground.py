import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({
    'study_hours': [2,3,4,5,6,7,8,9,10,11],
    'sleep_hours': [8,7.5,7,7,6.5,6,6,5.5,5,5],
    'exam_scores': [45,50,55,60,65,70,78,82,88,92]
    })

numeric_df = df[['study_hours', 'sleep_hours', 'exam_scores']]

corr_matrix = numeric_df.corr(method='pearson')
# print(corr_matrix)

plt.figure(figsize=(6,4))

sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1, vmax=1
        )

plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
