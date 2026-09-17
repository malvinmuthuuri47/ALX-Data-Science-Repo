"""
This module introduces the concept of KFold in sklearn to the learner
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

students = pd.DataFrame({
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Score': [15,22,30,35,48,55,63,71,82,91]
})

X = students[['Hours']]
y = students['Score']

kf = KFold(
    n_splits=5,
    shuffle=False
    # random_state=42
)

# for train_idx, test_idx in kf.split(students):
#     print('Train idx: ', train_idx)
#     print('Test idx:', test_idx)
#     print()

model = DecisionTreeRegressor(max_depth=3)

rmse_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    rmse_scores.append(rmse)

    print(f"Fold {fold}")
    print("Train indices:", train_idx)
    print("Test indices:", test_idx)
    print("RMSE:", round(rmse, 2))
    print()

print("Average Performance: ", np.mean(rmse_scores))