import numpy as np
import sklearn as sk
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# Get the source matrix
history = pd.read_excel('history.xlsx')
del history['Time']
history_norm = (history - history.mean()) / (history.max() - history.min())

X = history.as_matrix()
predict =pd.read_excel('predict.xlsx')
del predict['Time']
Y = predict.as_matrix().reshape(-1)

# Get the training data and the test data
X_train, X_test = X[:3300], X[3300:]
y_train, y_test = Y[:3300], Y[3300:]


tags = [
        'TIPS-5Y', 'TIPS-10Y', 'TIPS-20Y', 'TIPS-30Y', 'TIPS-LONG', 
        'UST BILL 1-Y RETURN', 'UST BILL 10-Y RETURN', 'SP500', 
        'LIBOR-OVERNIGHT', 'COMEX-NC-LONG', 'COMEX-NC-NET', 'SPDR-t',
        'USD/CNY'
        ]


pairs_single = {
        tags[0]: (0,), tags[1]: (1,), tags[2]: (2,), tags[3]: (3,), tags[4]: (4,), 
        tags[5]: (5,), tags[6]: (6,), tags[7]: (7,), tags[8]: (8,), tags[9]: (9,), 
        tags[10]: (10,), tags[11]: (11,), tags[12]: (12,)
        }
pairs_compose = {
        # '+'.join([tags[0], tags[7]]): (0, 7),
        # '+'.join([tags[0], tags[7]]): (0, 7),
        # '+'.join([tags[0], tags[5], tags[7]]): (0, 5, 7),
        # '+'.join([tags[0], tags[5], tags[7], tags[12]]): (0, 5, 7, 12),
        '+'.join([tags[0], tags[5], tags[7], tags[9], tags[12]]): (0, 5, 7, 9, 12)
        # '+'.join(tags): tuple(i for i in range(len(tags)))
        }

def predict(name, idx, epoch, lr):
    X_train_tmp, X_test_tmp = X_train[:, idx], X_test[:, idx]
    est = GradientBoostingRegressor(n_estimators=epoch, learning_rate=lr, 
            max_depth=5, random_state=0, loss='ls').fit(X_train_tmp, y_train)
    loss = mean_squared_error(y_test, est.predict(X_test_tmp)) 
    print('{} L2 loss: {}'.format(name, loss))

# for key in pairs_single:
#     predict(key, pairs_single[key], 10000, 0.1)

for key in pairs_compose:
    predict(key, pairs_compose[key], 200000, 0.01)
