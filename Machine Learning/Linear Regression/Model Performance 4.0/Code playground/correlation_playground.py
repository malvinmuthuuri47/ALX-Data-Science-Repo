import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

study_hours = [2,3,4,5,6,7,8,9,10,11]
exam_scores = [45,50,55,60,65,70,78,82,88,92]

df = pd.DataFrame({
    'study_hours': study_hours,
    'exam_scores': exam_scores
})

print(df)

from scipy.stats import pearsonr

r, p_value = pearsonr(study_hours, exam_scores)

print(f'Pearson r: {r:.4f}')
print(f'P-value: {p_value:.4f}')

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
        study_hours, exam_scores,
        color='steelblue',
        s=100,
        alpha=0.8,
        edgecolors='white',
        linewidth=1.5,
        label='Data Points',
        zorder=5
    )

# Trend line
m, b = np.polyfit(study_hours, exam_scores, 1)
x_line = np.array(study_hours)
ax.plot(
        x_line, x_line * m + b,
        color='red',
        linewidth=2,
        linestyle='--',
        label=f'Trend Line (r = {r:.4f})'
        )

ax.set_title(
        "Pearson's Correlation - Study Hours vs Exam Scores",
        fontsize=14, fontweight='bold'
        )
ax.set_xlabel('Study Hours', fontsize=12)
ax.set_ylabel('Exam Score', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
