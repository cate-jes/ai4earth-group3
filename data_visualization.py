
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import os
path = #yourpathhere
dir_list = os.listdir(path)
#print("Files and directories in '", path, "' :")
# prints all files
#print(dir_list)

data_f = #yourpathhere

gridMET = pd.read_csv(f"{data_f}/gridMET_area_weighted.csv")

# This file contains information of amount of water releases in a type of reservoir. We will need to combine this with GridMET to obtain
# our final input drivers.
reservoir_release = pd.read_csv(f"{data_f}/reservoir_releases_total.csv")

# This file contains simulated stream water temperatures obtained through a process-based model. In the paper these are used as outputs
# for model pretraining. Here, we will use these values to fill in any missing values from the real-world ground truth data.
dwallin_temp = pd.read_csv(f"{data_f}/dwallin_stream_preds.csv")

# The real-world stream temperature values are contained
target_temp = pd.read_csv(f"{data_f}/temperature_observations_forecast_sites.csv")
#site breakdown
sites = {
    1573: ["Cannonsville", "Pepacton"],
    1571: ["Cannonsville"],
    1565: ["Cannonsville"],
    1450: ["Pepacton"],
    1641: ["Neversink"]
    }

# Additionally, we are provided with the time split for model training and forecast
pretraining_time = { "start": '1985-05-01', "end": '2020-04-01'}
finetuning_time = { "start": '1985-05-01', "end": '2021-04-14'}
forecasting_time = {"start": '2021-04-16', "end": '2021-09-30'}

gridMET = gridMET[gridMET["seg_id_nat"].isin(sites.keys())] #narrowing down gridMET to only containing seg_id that matches
target_temp_ = target_temp[target_temp["seg_id_nat"].isin(sites.keys())]
dwallin_temp = dwallin_temp[dwallin_temp["seg_id_nat"].isin(sites.keys())]

# Ensuring all dates are of the same type
gridMET = gridMET.rename(columns={"time": "date"})
gridMET["date"] = pd.to_datetime(gridMET["date"])
dwallin_temp["date"] = pd.to_datetime(dwallin_temp["date"])
target_temp_["date"] = pd.to_datetime(target_temp_["date"]).dt.tz_localize(None)
reservoir_release["date"] = pd.to_datetime(reservoir_release["date"])

# Converting values from fahrenheit to celsius
gridMET["tmin"] = (gridMET["tmin"] -32) * (5/9)
gridMET["tmax"] = (gridMET["tmax"] - 32) * (5/9)

#-------VISUALIZATION/DATA UNDERSTANDING--------#
print("\n Number of values in dwallin temp:")
print(len(dwallin_temp));
print("\n Number of NA values in dwallin temp:")
print(dwallin_temp.isna().sum())
print("\n Number of values in target temp:")
print(len(target_temp_));
print("\n Number of NA values in target temp:")
print(target_temp_.isna().sum())
#------ 182 missing values for mean temp--------#
print("\nNumber of NA values in reservoir release:")
print(reservoir_release.isnull().sum())
#looks like no missing values in reservoir release, or dwallin
#---- check how many values-----
print("how much data we got:", len(gridMET))
#print("\n Counts per year:\n", target_temp_.index.year.value_counts());
#------visualize our data------# subsection....
dwallin_temp.plot(x="date", y="dwallin_temp_c")
target_temp_.plot(x="date", y="mean_temp_c")
#a very high outlier with resevoir release
reservoir_release.plot(x="date", y="release_volume_cms")

