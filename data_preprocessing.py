
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

#------CODE TO MERGE DATASETS + VISUALIZE ------#
#Goal: fill in missing data in target temp using dwallin simulated temps, match by site id and date. 
test_target_temp = target_temp_.copy()
test_dwallin = dwallin_temp.copy()
print("\n MERGED DATA------------");
merged_df = pd.merge(target_temp_, dwallin_temp, on=["date", "seg_id_nat"], how="left") # is this the best type of merge for this data?
print(merged_df.head())
#so far this looks the closest.
merged_df["mean_temp_c"] = np.where(merged_df["mean_temp_c"].isna(), merged_df["dwallin_temp_c"], merged_df["mean_temp_c"])
print(merged_df.head())
#get rid of nonexistent sites:
merged_df = merged_df.dropna(subset=['site_id'])
print(merged_df.head())
#now let's check the merged data:
print(merged_df.isna().sum());

print("\n Merged dataset number of rows:");
print(len(merged_df));
#merged_df[["mean_temp_c", "dwallin_temp_c"]].plot();
merged_df.plot(x="date", y=["mean_temp_c", "dwallin_temp_c"]);
merged_df.plot(x="date", y="mean_temp_c");

"""
Current questions: 
is a left merge the best kind of merge for this dataframe? 
/what could be causing so many NA values
how to interpret noncontinuity in the graph?
"""