import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({
    'study_hours': [2,3,4,5,6,7,8,9,10,11],
    'sleep_hours': [8,7,7,6,6,7,8,7,6,5],
    'exam_scores': [45,50,55,60,65,70,78,82,88,92],
    'stress_level': [8,7,6,6,5,4,4,3,3,2]
    })

corr_matrix = df.corr()

print(corr_matrix.round(3))

fig,ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor='white',
        ax=ax
        )

ax.set_title(
        'Correlation Matrix - Student Performance',
        fontsize=14, fontweight='bold', pad=15
        )

plt.tight_layout()
plt.show()
