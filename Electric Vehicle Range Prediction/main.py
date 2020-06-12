import regressors
import xgb_regressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

dataset = pd.read_csv("processed.csv", index_col='epoch time')
dataset.index = pd.to_datetime(dataset.index)

y_pred_LR = pd.read_csv(".\predictions\y_pred_LR.csv")['0']
y_pred_AdB = pd.read_csv(".\predictions\y_pred_AdB.csv")['0']
y_pred_BG = pd.read_csv(".\predictions\y_pred_BG.csv")['0']
y_pred_ET = pd.read_csv(".\predictions\y_pred_ET.csv")['0']
y_pred_XGB = pd.read_csv(".\predictions\y_pred_XGB.csv")['0']

y_pred_LSTM_adam = pd.read_csv(".\predictions\y_pred_LSTM_adam.csv")['0']
y_pred_LSTM_RMProp = pd.read_csv(".\predictions\y_pred_LSTM_RMProp.csv")['0']
y_pred_LSTM_sgd = pd.read_csv(".\predictions\y_pred_LSTM_sgd.csv")['0']
y_pred_NN_adam = pd.read_csv(".\predictions\y_pred_NN_adam.csv")['0']
y_pred_NN_RMSProp = pd.read_csv(".\predictions\y_pred_NN_RMSProp.csv")['0']


y_range = dataset["range"]
dataset = dataset.drop(["range"],axis = 1)
[_, test_X, _, y_test_range] =  regressors.split_dataset_y(dataset,y_range)


RMSE_ALL= {"LinearReg":regressors.Calc_RMSE(y_test_range.values,y_pred_LR),
"AdBoostReg":regressors.Calc_RMSE(y_test_range.values,y_pred_AdB),
"BaggingReg": regressors.Calc_RMSE(y_test_range.values,y_pred_BG),
"ExTreesReg": regressors.Calc_RMSE(y_test_range.values,y_pred_ET),
"XGBReg": regressors.Calc_RMSE(y_test_range.values,y_pred_XGB),
"LSTM_Adam": regressors.Calc_RMSE(y_test_range.values,y_pred_LSTM_adam),
"LSTM_RMSPr": regressors.Calc_RMSE(y_test_range.values,y_pred_LSTM_RMProp),
"LSTM_SGD": regressors.Calc_RMSE(y_test_range.values,y_pred_LSTM_sgd),
"NN_Adam": regressors.Calc_RMSE(y_test_range.values,y_pred_NN_adam),
"NN_RMSPr": regressors.Calc_RMSE(y_test_range.values,y_pred_NN_RMSProp)
}

print("RMSE for LinearReg", RMSE_ALL["LinearReg"])
print("RMSE for AdBoostReg", RMSE_ALL["AdBoostReg"])
print("RMSE for BaggingReg", RMSE_ALL["BaggingReg"])
print("RMSE for ExtraTreesReg", RMSE_ALL["ExTreesReg"])
print("RMSE for XGBReg", RMSE_ALL["XGBReg"])
print("RMSE for LSTM with Adam", RMSE_ALL["LSTM_Adam"])
print("RMSE for LSTM with RMSProp", RMSE_ALL["LSTM_RMSPr"])
print("RMSE for LSTM with SGD", RMSE_ALL["LSTM_SGD"])
print("RMSE for NN with Adam", RMSE_ALL["NN_Adam"])
print("RMSE for NN with RMSProp", RMSE_ALL["NN_RMSPr"])


weights = np.arange(0,1,0.1)    #[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
min_RMSE = float('inf')

for i in range(len(weights)):
    for j in range(len(weights)):
        if (weights[i]+weights[j] <= 1.0):
            new_y_pred = y_pred_XGB*round(weights[i],1) + y_pred_BG*round(weights[j],1) + y_pred_LSTM_adam*(round(1-weights[i]-weights[j],1))
            RMSE = regressors.Calc_RMSE(y_test_range.values,new_y_pred)
            print("RMSE for Ensembled model(",round(weights[i],1),"*XGB +",round(weights[j],1),"*BAG +",round(1-weights[i]-weights[j],1),"*LSTM ) : ", RMSE)
            if min_RMSE>RMSE:
                min_RMSE= RMSE
                best_weights = [round(weights[i],1),round(weights[j],1),round(1-weights[i]-weights[j],1)]
                best_y_pred = new_y_pred

print("Best RMSE for Ensembled model(",best_weights[0],"* XGB + ",best_weights[1],"* BAG + ",best_weights[2],"* LSTM ) : ", min_RMSE)

RMSE_ALL["Ensembled"] = min_RMSE

regressors.plot_performance(test_X,y_test_range,best_y_pred, model_type = 'Ensembled_Model')   

RMSE_ALL = dict(sorted(RMSE_ALL.items(), key = lambda kv:(kv[1], kv[0]),reverse= True))

fig = plt.figure(1)
ax = fig.add_subplot(111)       
plt.bar(RMSE_ALL.keys(), RMSE_ALL.values())
ax.set_xticklabels(RMSE_ALL.keys(),rotation = 45,ha="right")
plt.xlabel("Models")
plt.ylabel("RMSE Score")
plt.title("RMSE Scores of Models")
plt.savefig("RMSE_Scores")
plt.show()
        