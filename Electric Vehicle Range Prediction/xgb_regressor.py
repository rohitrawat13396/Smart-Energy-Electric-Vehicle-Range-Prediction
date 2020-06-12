'''
Author : @virajk
Description : Models that will be used to predict EV range
We will try to iteratively tune the hyperparameters
'''

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from regressors import *

'''
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(9,12)
    for min_child_weight in range(5,8)
]
# Define initial best params and MAE
min_rmse = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=999,
        seed=42,
        nfold=5,
        metrics={'rmse'},
        early_stopping_rounds=10
    )
    # Update best MAE
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (max_depth,min_child_weight)


    print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))
'''

'''
Best params: 9, 6, RMSE: 58.1229112
'''
'''
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]

min_rmse = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=999,
        seed=42,
        nfold=5,
        metrics={'rmse'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (subsample,colsample)
print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))
'''

'''
Best params: 0.9, 0.9, RMSE: 56.0651474
'''

#%time
'''
min_rmse = float("Inf")
best_params = None

for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta
    # Run and time CV
    #%time
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=999,
            seed=42,
            nfold=5,
            metrics=['rmse'],
            early_stopping_rounds=10
          )
    # Update best score
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\trmse {} for {} rounds\n".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = eta
    print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
'''
'''
Best params: 0.01, RMSE: 55.9398066
'''


#xGBoost#xg_reg.fit(X_train,y_train)
#preds = xg_reg.predict(X_test)


#values = pd.DataFrame( zip(list(y_test.index) , list(y_test) , list(preds)  ) , columns=['time','range','preds'] )
#values.set_index('time')


#print(mean_squared_error(list(y_test),list(preds)))
#values.plot(figsize=(10,5), grid=True)
#plt.show()




global preds
global y_test

def xgb_model(data) :
    x, y = data.iloc[:, 0:data.shape[1] - 1], data.iloc[:, data.shape[1] - 1]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123,shuffle=False)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
    'colsample_bytree': 0.7,
     'eta': 0.01,
     'eval_metric': 'rmse',
     'max_depth': 9,
     'min_child_weight': 7,
     'objective': 'reg:linear',
     'subsample': 0.9}
    params['eval_metric'] = "rmse"

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=999,
        evals=[(dtest, "Test")],
        early_stopping_rounds=10,
        verbose_eval=50

    )

    num_boost_round = model.best_iteration + 1
    best_model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        verbose_eval=50
    )

    preds = best_model.predict(dtest)
    values = pd.DataFrame( zip(list(y_test.index) , list(y_test) , list(preds)  ) , columns=['time','range','preds'] )
    values.set_index('time')
    plot_performance(y_test,list(y_test),list(preds),"xgb")

    xgb.plot_importance(best_model)
    plt.show()


    RMSE = rmse(y_test,preds)
    return RMSE, list(preds)



def rmse(y_test,preds) :

    rmse = math.sqrt(mean_squared_error(list(y_test), list(preds)))

    print("Root Mean squared error for XGB={}".format(str(rmse)))

    return rmse

'''
def y_pred() :
    return list(preds)
'''

if __name__ == "__main__" :
    data = pd.read_csv("processed.csv", index_col='epoch time')
    data.index = pd.to_datetime(data.index)
    RMSE_XGB,y_pred_XGB = xgb_model(data)
    pd.DataFrame(y_pred_XGB).to_csv(".\predictions\y_pred_XGB.csv", index = False)
    