# Author: Luo Xuan
# This file generate the analysis graph in the doc
# Run: python3 graph.py

import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

history = pd.read_excel('history.xlsx')
predict =pd.read_excel('predict.xlsx')

df = history.copy()
del df['Time']
df_norm = (df - df.mean()) / (df.max() - df.min())
df_norm.plot(figsize=(20, 10), title='All related feature normalized figure')
plt.savefig('f1.png')
plt.show()

f, ax = plt.subplots(4, 4, sharex=True, figsize=(20,10))
predict.plot(title="X benchmark trend Unit: x/d", ax=ax[0][0], color=tuple(tuple(e) for e in np.random.rand(1, 3).tolist()))
history['TIPS-5Y'].plot(title='history TIPS-5Y', ax=ax[0][1], color=((0.1, 0.2, 0.3)))
history['TIPS-10Y'].plot(title='history TIPS-10Y', ax=ax[0][2], color=((0.2, 0.1, 0.9)))
history['TIPS-20Y'].plot(title='history TIPS-20Y', ax=ax[0][3], color=((0.1, 0.27, 0.2)))
history['TIPS-30Y'].plot(title='history TIPS-30Y', ax=ax[1][0], color=((0.1, 0.6, 0.3)))
history['TIPS-LONG'].plot(title='history TIPS-LONG', ax=ax[1][1], color=((0.6, 0.2, 0.3)))
history['UST Bill 1-Y RETURN'].plot(title='history UST Bill 1-Y RETURN', ax=ax[1][2], color=((0.6, 0.7, 0.3)))
history['UST BILL 10-Y RETURN'].plot(title='history UST BILL 10-Y RETURN', ax=ax[1][3], color=((0.8, 0.4, 0.3)))
history['SP500'].plot(title='history SP500', ax=ax[2][0], color=((0.1, 0.6, 0.3)))
history['LIBOR-OVERNIGHT'].plot(title='history LIBOR-OVERNIGHT', ax=ax[2][1], color=((0.2, 0.2, 0.65)))
history['COMEX-NC-LONG'].plot(title='history COMEX-NC-LONG', ax=ax[2][2], color=((0.1, 0.62, 0.32)))
history['COMEX-NC-SHORT'].plot(title='history COMEX-NC-NET', ax=ax[2][3], color=((0.71, 0.12, 0.3)))
history['SPDR:t'].plot(title='history SPDR:t', ax=ax[3][0], color=((0.81, 0.12, 0.43)))
history['USD/CNY'].plot(title='USD/CNY', ax=ax[3][1], color=((0.1, 0.62, 0.3)))

plt.savefig('f2.png')
plt.show()
