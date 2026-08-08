import numpy as np

# scores = np.array([65, 82, 91, 73, 95])

# print(np.where(scores > 80))

scores = np.array([
    [65, 82, 91],
    [73, 95, 60],
    [88, 71, 55]
    ])

r, c = np.where(scores > 80)
print(f'{r}\n, {c}')
