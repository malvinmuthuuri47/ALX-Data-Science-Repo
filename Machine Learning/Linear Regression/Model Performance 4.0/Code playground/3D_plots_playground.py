import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([2,4,5,4,5,7,8,9,10,12])
z = np.array([3,5,4,6,7,8,9,10,11,13])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot a 3D scatter
ax.scatter(
        x, y, z,
        color='steelblue',
        s=100,
        alpha=0.8,
        edgecolors='white',
        linewidths=1.5
        )

# Label all three axes
ax.set_xlabel('X Axis', fontsize=12)
ax.set_ylabel('Y Axis', fontsize=12)
ax.set_zlabel('Z Axis', fontsize=12)

ax.set_title('3D Scatter Plot', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
