#!/usr/bin/env python
# coding: utf-8

# In[61]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[62]:


dataset = read_csv('processed.csv', header=0, index_col=0)


# In[ ]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    summ =  concat(cols, axis=1)
    summ.columns = names
    if dropnan:
        summ.dropna(inplace=True)
    return summ
def LSTM_ML(dataset):
    dataset = read_csv('processed.csv', header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)
    col = ['var1(t)', 'var2(t)', 'var3(t)', 'var4(t)', 'var5(t)',
           'var6(t)', 'var7(t)', 'var8(t)', 'var9(t)', 'var10(t)', 'var11(t)',
           'var12(t)', 'var13(t)', 'var14(t)', 'var15(t)', 'var16(t)', 'var17(t)',
           'var18(t)', 'var19(t)', 'var20(t)', 'var21(t)', 'var22(t)', 'var23(t)',
           'var24(t)', 'var25(t)']
    reframed.drop(col, axis=1, inplace=True)
    values = reframed.values
    train = values[:-1473, :]
    test = values[-1473:, :]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    pyplot.plot(inv_y, label='Actual')
    pyplot.plot(inv_yhat, label='Predicted')
    pyplot.legend()
    pyplot.savefig('temp2.png',dpi=300)
    return(model,inv_yhat,rmse)
model,inv_yhat,rmse = LSTM_ML(dataset)
#with open('y_pred_LSTM.csv', 'w') as f:
#    for item in inv_yhat:
#            f.write("%s\n" % item)
pd.DataFrame(inv_yhat).to_csv("y_pred_LSTM.csv", index = False)


# In[45]:


###########################################################################
###########################################################################


# In[49]:


from keras.callbacks import EarlyStopping
def neural(dataset):
    train_X = dataset.drop(columns=['range'])
    train_y = dataset[['range']]
    train_X,test_X = train_X[:-1473][:],train_X[-1473:][:]
    train_y,test_y = train_y[:-1473][:],train_y[-1473:][:]
    from keras.layers import Dropout
    n_cols = train_X.shape[1]
    model_mc = Sequential()
    model_mc.add(Dense(200, activation='relu', input_shape=(n_cols,)))
    model_mc.add(Dropout(0.2))
    model_mc.add(Dense(200, activation='relu'))
    model_mc.add(Dense(200, activation='relu'))
    model_mc.add(Dense(1))
    model_mc.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping_monitor = EarlyStopping(patience=3)
    model_mc.fit(train_X, train_y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])
    y_pred = model_mc.predict(test_X)
    pyplot.plot(test_y, label='Actual')
    pyplot.plot(y_pred, label='Predicted')
    pyplot.legend()
    pyplot.savefig('temp3.png',dpi=300)
    rmse = sqrt(mean_squared_error(y_pred, test_y))
    print('Test RMSE: %.3f' % rmse)
    return(model_mc,y_pred,rmse)
model,y_pred,rmse=neural(dataset)
#with open('y_pred_NN.csv', 'w') as f:
#    for item in y_pred:
#        for x in item:
#            f.write("%s\n" % x)
pd.DataFrame(y_pred).to_csv("y_pred_NN.csv", index = False)


# In[ ]:




