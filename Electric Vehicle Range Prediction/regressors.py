import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("processed.csv", index_col='epoch time')
dataset.index = pd.to_datetime(dataset.index)

def plot_performance(test, y_test, pred, model_type):

    test = pd.DataFrame(test)
    plt.figure(figsize=(15,3))
    plt.title('Plot predictions vs true value for '+ model_type)
    plt.xlabel('epoch time')
    plt.ylabel('range')
    plt.plot(test.index, y_test, label='True data')
    plt.plot(test.index, pred, label='Predicted data')
    plt.legend()
    plt.xlim(left=test.index[250], right=test.index[570])
    plt.savefig(model_type)
    plt.show()
    

def Calc_RMSE(true,pred):
  #RMSE = np.sqrt(np.sum((true - pred)**2)/len(true))
  RMSE = sqrt(mean_squared_error(true,pred))
  return RMSE

def split_dataset_y(dataset,y):
  train, test, y_train, y_test = train_test_split(dataset,y, test_size = 0.20, shuffle=False)
  return [train, test, y_train, y_test]

def prediction(reg,data,true):
  pred = reg.predict(data)
  return [Calc_RMSE(true,pred),pred]

def linear_reg(dataset):
  
  y_range = dataset["range"]
  dataset = dataset.drop(["range"],axis = 1)
  [train_X, test_X, y_train_range, y_test_range] =  split_dataset_y(dataset,y_range)
  model = LinearRegression().fit(train_X,y_train_range)
  RMSE,y_pred = prediction(model,test_X,y_test_range)
  print("Root Mean squared error for LR=", RMSE) 
  model_type = 'linear_Regression_model' 
  plot_performance(test_X, y_test_range, y_pred, model_type)

  return [model,RMSE, y_pred]

def AdaBoostRegressor_model(dataset):

  y_range = dataset["range"]
  dataset = dataset.drop(["range"],axis = 1)
  [train_X, test_X, y_train_range, y_test_range] =  split_dataset_y(dataset,y_range)
  min_RMSE = float('inf')
  best_model=[]
  for i in range(5):
    model = AdaBoostRegressor(random_state=None).fit(train_X,y_train_range)
    RMSE,y_pred = prediction(model,test_X,y_test_range)
    if min_RMSE>RMSE:
      min_RMSE = RMSE
      best_model = model
      best_y_pred= y_pred
  print("Root Mean squared error for AdaBoost Regressor=", min_RMSE)
  model_type = 'AdaBoost_Regressor_model' 
  plot_performance(test_X, y_test_range, best_y_pred, model_type)

  return [best_model,min_RMSE,best_y_pred]

def BaggingRegressor_model(dataset):

  y_range = dataset["range"]
  dataset = dataset.drop(["range"],axis = 1)
  [train_X, test_X, y_train_range, y_test_range] =  split_dataset_y(dataset,y_range)
  min_RMSE = float('inf')
  best_model=[]
  best_y_pred=[]
  for i in range(5):
    model = BaggingRegressor(random_state=None).fit(train_X,y_train_range)
    RMSE,y_pred = prediction(model,test_X,y_test_range)
    if min_RMSE>RMSE:
      min_RMSE = RMSE
      best_model = model
      best_y_pred= y_pred
  print("Root Mean squared error for BaggingRegressor_model=", min_RMSE)
  model_type = 'Bagging_Regressor_model' 
  plot_performance(test_X, y_test_range, best_y_pred, model_type)

  return [best_model,min_RMSE,best_y_pred]

def ExtraTreesRegressor_model(dataset):

  y_range = dataset["range"]
  dataset = dataset.drop(["range"],axis = 1)
  [train_X, test_X, y_train_range, y_test_range] =  split_dataset_y(dataset,y_range)
  min_RMSE = float('inf')
  best_model=[]
  best_y_pred=[]
  for i in range(5):
    model = ExtraTreesRegressor(random_state=None).fit(train_X,y_train_range)
    RMSE,y_pred = prediction(model,test_X,y_test_range)
    if min_RMSE>RMSE:
      min_RMSE = RMSE
      best_model = model
      best_y_pred= y_pred
  print("Root Mean squared error for ExtraTreesRegressor_model=", min_RMSE)
  model_type = 'Extra-Trees_Regressor_model' 
  plot_performance(test_X, y_test_range, best_y_pred, model_type)

  return [best_model,min_RMSE,best_y_pred]
if __name__ == "__main__" :
  _,RMSE_LR,y_pred_LR = linear_reg(dataset)
  _,RMSE_AdB,y_pred_AdB = AdaBoostRegressor_model(dataset)
  _,RMSE_BG,y_pred_BG = BaggingRegressor_model(dataset)
  _,RMSE_ET,y_pred_ET = ExtraTreesRegressor_model(dataset)

  pd.DataFrame(y_pred_LR).to_csv(".\predictions\y_pred_LR.csv", index = False)
  pd.DataFrame(y_pred_AdB).to_csv(".\predictions\y_pred_AdB.csv", index = False)
  pd.DataFrame(y_pred_BG).to_csv(".\predictions\y_pred_BG.csv", index = False)
  pd.DataFrame(y_pred_ET).to_csv(".\predictions\y_pred_ET.csv", index = False)


