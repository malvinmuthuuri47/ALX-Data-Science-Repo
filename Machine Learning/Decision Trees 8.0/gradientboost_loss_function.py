"""
This module is intended to show the learner how different loss values in the loss parameter of the
GradientBoostRegressor model affect the overall model prediction
"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'house_size': [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240],
    'house_price': [150000,180000,210000,240000,265000,300000,320000,355000,375000,410000,440000,465000,495000,520000,550000,580000,605000,635000,660000,690000],

})

X = df[['house_size']]
y = df['house_price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

'''Comparing different loss functions'''
loss_functions = ['squared_error', 'absolute_error', 'huber']

print(f'{"Loss Function":<20} {"R²":>10} {"RMSE":>12}')
print('=' * 45)

for loss in loss_functions:
    model = GradientBoostingRegressor(
        loss=loss,
        n_estimators=100,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'{loss:<20} {r2:>10.4f} {rmse:>12.2f}')