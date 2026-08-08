import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Two numpy arrays
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([3,5,4,7,8,9,10,12,11,14])

print(x.shape)
x = x.reshape(-1, 1)
print(x.shape)

model = LinearRegression()
model.fit(x, y)

slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(x, y)

print(f'Slope (m): {slope:.4f}')
print(f'Intercept (c): {intercept:.4f}')
print(f'R2 Score: {r_squared:.4f}')

# predict y for new x values
new_x = np.array([[11], [12], [15]])
predictions = model.predict(new_x)

print("Value of Y for the new x values: ", predictions)

y_predicted = model.predict(x)

fig, ax = plt.subplots(figsize=(10,5))

# Scatter the original data points
ax.scatter(
        x, y,
        color='steelblue',
        s=80,
        label='Actual Data',
        zorder=5
        )

# Draw the regression line
ax.plot(
        x, y_predicted,
        color='red',
        linewidth=2,
        label=f'Regression Line (R²={r_squared:.4f})'
        )

# Annotate the equation on the chart
ax.annotate(
        f'y = {slope:.2f}x + {intercept:.2f}',
        xy=(2, 12),
        fontsize=12,
        color='red'
        )

ax.set_title('Linear Regression', fontsize=14, fontweight='bold')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
