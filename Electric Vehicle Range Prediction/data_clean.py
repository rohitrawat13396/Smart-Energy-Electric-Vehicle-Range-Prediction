import pandas as pd
import os
from datetime import datetime
import sys
import numpy as np
import time
from pandas_profiling import ProfileReport

dir = "\dataset\LEAF Data Logs 1\\"
files = os.listdir(os.getcwd()+dir)
import matplotlib.pyplot as plt
data_files = []

#    'Date/Time',

important_columns = [
    "Elv",
    "Speed",
    "Gids",
    "SOC",
    "AHr",
    "Pack Volts",
    "Pack Amps",
    "CP mV Diff",
    "Pack T1 C",
    "12v Bat Amps",
    "Hx",
    "Odo(km)",
    "TP-FL",
    "TP-FR",
    "TP-RR",
    "TP-RL",
    "Ambient",
    "SOH",
    "epoch time",
    "Motor Pwr(w)",
    "Aux Pwr(100w)",
    "A/C Pwr(250w)",
    "Est Pwr A/C(50w)",
    "Est Pwr Htr(250w)",
    "Motor Temp",
    "Torque Nm"
]

#
dataframe = pd.DataFrame()
for file in files :
    if file.endswith(".csv") :
        data = pd.read_csv(os.getcwd()+dir+file)
        #try :
        #    data["epoch time"] = data['epoch time'].apply(lambda x : datetime.timestamp(datetime.strptime(x,")))
        #except Exception as e :
        #    data["epoch time"] = data['epoch time'].apply(lambda x : datetime.timestamp(datetime.strptime(x,"%d/%m/%Y %H:%M:%S")))

        #data["epoch time"] = data['Date/Time'].apply(lambda x : datetime.strptime(x,"%m/%d/%Y %H:%M"))


        data["epoch time"] = data['epoch time'].apply(lambda x : datetime.fromtimestamp(x))

        #.apply(lambda x : datetime.fromtimestamp(x))
        data['Gids'] = data['Gids'].apply(lambda x: int(x) * 80)
        data['SOC'] = data['SOC'].apply(lambda x: float(x) / 10000)  # Should yield values below 100 - outliers present
        data['AHr'] = data['AHr'].apply(lambda x: float(x) / 10000)
        data['energy_level'] = data['Gids'] * data['SOC'] * 360  # New column created
        # data["HX"] -  how to handle this ?
        # Tire Pressure is empty in all cases

        data['Ambient'] = (data['Ambient'] - 32) * 5 / 9  # Converting to clecius
        data['Motor Pwr(w)'] = data['Motor Pwr(w)']
        data['Aux Pwr(100w)'] = data['Aux Pwr(100w)'] * 100
        data["A/C Pwr(250w)"] = data["A/C Pwr(250w)"] * 250
        data["Est Pwr Htr(250w)"] = data["Est Pwr Htr(250w)"] * 250
        data["Motor Temp"] = data["Motor Temp"] - 40
        data["Torque Nm"] - abs(data["Torque Nm"])
        data = data[important_columns]
        data = data.set_index('epoch time')
        data = data.sort_values(by=['epoch time'], ascending=True)

        #data['timestep'] = data.index
        #data['timestep'] = data['timestep'].apply(lambda x : int( time.mktime(x.timetuple()) ))


        data = data.resample("1T",).mean()
        #data = data.fillna(method="ffill")
        data = data.dropna()
        data = data[data['SOC'] <= 100]
        distance_travelled = (data["Odo(km)"] - data["Odo(km)"].shift(1)).values
        charge_drop = (data["SOC"] - data["SOC"].shift(1)).values
        soc = data['SOC'].values
        distance_travelled[0] = 0
        charge_drop[0] = 0
        mil_range = []
        #data['timestep2'] = data['timestep'].apply(lambda x: datetime.fromtimestamp(x))

        data['driver_aggression'] = 0
        data['Percentile_rank'] = data["Speed"].rank(pct=True)

        for index, row in data.iterrows():
            if row['Percentile_rank'] > 0.9:
                data.loc[index, 'driver_aggression'] = 10
            elif row['Percentile_rank'] > 0.7:
                data.loc[index, 'driver_aggression'] = 7
            elif row['Percentile_rank'] > 0.5:
                data.loc[index, 'driver_aggression'] = 5


        default_eff = 1.60935 #https://www.reddit.com/r/leaf/comments/a174gt/2018_owners_what_is_your_highway_range/
        for x in range( 0, len(distance_travelled) ) :
         if distance_travelled[x] == 0 or charge_drop[x] == 0 :
             mil_range.append(default_eff*soc[x])           #Average known mileage of Nissan Leaf
         else :
             if soc[x] <= soc[x-1]:         #What if car was charged
                mil_range.append((distance_travelled[x]/abs(charge_drop[x]))*soc[x])
             else :
                 mil_range.append(default_eff* soc[x])  # Average known mileage of Nissan Leaf
        data['range'] = mil_range
        dataframe = dataframe.append(data)



#Sort
data = dataframe
data = data[ data['range']  <= 500 ]
data = data[ data['range']  > 0 ]



data = data.drop(columns=['Percentile_rank'])
data.loc[:,:].to_csv("processed.csv")




print(data.shape)

profile = ProfileReport(data, title='Pandas Profiling Report', html={'style':{'full_width':True}})
profile.to_file("output.html")

