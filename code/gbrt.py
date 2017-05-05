import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

history = pd.read_excel('history.xlsx')
X = history.as_matrix()[:, 1:]
predict =pd.read_excel('predict.xlsx')
Y = predict.as_matrix()[:, 1:].reshape(-1)
X_train, X_test = X[:3300], X[3300:]
y_train, y_test = Y[:3300], Y[3300:]
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
        max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
mean_squared_error(y_test, est.predict(X_test)) 
