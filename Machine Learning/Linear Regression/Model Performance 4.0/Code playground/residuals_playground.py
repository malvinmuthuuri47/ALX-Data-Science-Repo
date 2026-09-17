import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    'house_size': [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240],
    'house_price': [150000,180000,210000,240000,265000,300000,320000,355000,375000,410000,440000,465000,495000,520000,550000,580000,605000,635000,660000,690000]
    })

X = df[['house_size']]
y = df['house_price']

X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
        )

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

residuals = y_test - y_pred

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
        residuals,
        bins=10,
        color='steelblue',
        edgecolor='white',
        alpha=0.85,
        label='Residuals'
        )

ax.axvline(
        x=0,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Zero Line (perfect prediction)'
        )

ax.axvline(
        x=np.mean(residuals),
        color='orange',
        linestyle='-',
        linewidth=2,
        label=f'Mean Residual = {np.mean(residuals):.2f}'
        )

ax.set_title(
        'Histogram of Residuals - House Size vs House Price',
        fontsize=14,
        fontweight='bold',
        pad=15
        )

ax.set_xlabel('Residual (Actual Price - Predicted Price)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.4, axis='y')

plt.tight_layout()
plt.show()
