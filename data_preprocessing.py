
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
path = #...
dir_list = os.listdir(path)
#print("Files and directories in '", path, "' :")
# prints all files
#print(dir_list)

data_f = #....

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
#Fill in our missing data
test_target_temp = target_temp_.copy()
test_dwallin = dwallin_temp.copy()
print("\n MERGED DATA------------");
target_temp_filled = pd.merge(target_temp_, dwallin_temp, on=["date", "seg_id_nat"], how="left")
print(target_temp_filled.head())
#now that we've merged, replace missing values in mean_temp_c
target_temp_filled["mean_temp_c"] = np.where(target_temp_filled["mean_temp_c"].isna(), target_temp_filled["dwallin_temp_c"], target_temp_filled["mean_temp_c"])
#now let's check the merged data:
print(target_temp_filled.isna().sum());

test_target_temp.plot(x="date", y="mean_temp_c");
target_temp_filled.plot(x="date", y="mean_temp_c");
target_temp_filled.plot(x="date", y=["mean_temp_c", "dwallin_temp_c"]);
#--------SPLITTING BASED ON TIME-----------#
# Selecting sections of data based on times, and segregating it into pretraining and finetuning
gridMET_pretrain = gridMET[(gridMET['date'] >= pretraining_time["start"]) & (gridMET['date'] <= pretraining_time["end"])]
dwallin_temp = dwallin_temp[(dwallin_temp['date'] >= pretraining_time["start"]) & (dwallin_temp['date'] <= pretraining_time["end"])]

gridMET_finetune = gridMET[(gridMET['date'] >= finetuning_time["start"]) & (gridMET['date'] <= finetuning_time["end"])]
target_temp_filled = target_temp_filled[(target_temp_filled['date'] >= finetuning_time["start"]) & (target_temp_filled['date'] <= finetuning_time["end"])]

input_data_pretrain = pd.merge(gridMET_pretrain, dwallin_temp, on=["seg_id_nat", "date"], how="right")
print(input_data_pretrain.head());
print(len(input_data_pretrain));
input_data_finetune = pd.merge(gridMET_finetune, target_temp_filled, on=["seg_id_nat", "date"], how="right")
print(input_data_finetune.head());
#--------COMBINATION FUNCTION------#
def combine_reservoir_driver_data(input_data, reservoir_release, site, target_variable=None):
    # Extracting site-wise input
    input_ = input_data[input_data["seg_id_nat"] == site]

    # Combining with reservoir based on site type
    res_columns = []
    for site_type in sites[site]:
        print(site_type)
        input_ = input_.merge(reservoir_release[reservoir_release["reservoir"] == site_type][["release_volume_cms", "date"]], on="date", how="left")
        input_.rename(columns={"release_volume_cms": f"release_volume_cms_{site_type}"}, inplace=True)
        res_columns.append(f"release_volume_cms_{site_type}")
    # Preparing data as input/target
    site_input_X = input_[["tmin", "tmax", "srad", *res_columns]].copy()
    if target_variable:
        site_input_X["ar1"] = input_[target_variable].shift(1)
    site_input_Y = input_[target_variable]

    # Drop NaN due to shifting time for ar value
    site_input_X.drop(index=0, inplace=True)
    site_input_Y.drop(index=0, inplace=True)

    return site_input_X, site_input_Y, site_input_X.shape

site = 1573 # Change value based on what site you would like to preprocess data for!

pretrain_site_input_X, pretrain_site_input_Y, shape = combine_reservoir_driver_data(input_data_pretrain, reservoir_release, site, target_variable="dwallin_temp_c")


# Pretrain_site_input_X contains all our inputs, while pretrain_site_input_Y contains our output
# We can use the shape later as we tinker with our dataset/initialize our models
print(pretrain_site_input_X.head())
print(pretrain_site_input_Y.head())

finetune_site_input_X, finetune_site_input_Y, shape = combine_reservoir_driver_data(input_data_finetune, reservoir_release, site, target_variable="mean_temp_c")

#-------------Splitting data into train and test data-----------
pretrain_site_input_X_train, pretrain_site_input_X_test, pretrain_site_input_Y_train, pretrain_site_input_Y_test = train_test_split(pretrain_site_input_X, pretrain_site_input_Y, test_size=0.2, shuffle=False)
finetune_site_input_X_train, finetune_site_input_X_test, finetune_site_input_Y_train, finetune_site_input_Y_test = train_test_split(finetune_site_input_X, finetune_site_input_Y, test_size=0.2, shuffle=False)

#compute the mean and normalize using standardscaler.
x_scaler = StandardScaler()
y_scaler = StandardScaler()
#convert y to a numpy array
pretrain_site_input_Y_train = np.array(pretrain_site_input_Y_train)
pretrain_site_input_Y_test = np.array(pretrain_site_input_Y_test)
finetune_site_input_Y_train = np.array(finetune_site_input_Y_train)
finetune_site_input_Y_test = np.array(finetune_site_input_Y_test)

x_scaler.fit(finetune_site_input_X_train)
y_scaler.fit(finetune_site_input_Y_train.reshape(-1, 1))

pretrain_site_input_X_train = x_scaler.transform(pretrain_site_input_X_train)
pretrain_site_input_X_test = x_scaler.transform(pretrain_site_input_X_test)
finetune_site_input_X_train = x_scaler.transform(finetune_site_input_X_train)
finetune_site_input_X_test = x_scaler.transform(finetune_site_input_X_test)

new_pretrain_site_input_Y_train = y_scaler.transform(pretrain_site_input_Y_train.reshape(-1, 1))
new_spretrain_site_input_Y_test = y_scaler.transform(pretrain_site_input_Y_test.reshape(-1, 1))
new_finetune_site_input_Y_train = y_scaler.transform(finetune_site_input_Y_train.reshape(-1, 1))
new_finetune_site_input_Y_test = y_scaler.transform(finetune_site_input_Y_test.reshape(-1, 1))
#comes out as a numpy array of features for input --> unlabeled

#--------fill in missing values----------#
forecast = pd.read_csv(f"{data_f}/forecast_data_E0.csv")
forecast_reservoir = pd.read_csv(f"{data_f}/forecast_release_data.csv")
#replacing missing values and dropping unlabeled sites,
forecast["max_temp_c"] = forecast["max_temp_c"].interpolate()
forecast = forecast.dropna(subset=["site_id"]).copy()
#converting both to correct date time format
forecast["date"] = pd.to_datetime(forecast["date"])
forecast_reservoir['date'] = pd.to_datetime(forecast_reservoir['date'])
forecast.head()

#-------combine forecast data---------
#combine the forecast date with the forecast reservoir data --> For testing the model.
forecast_site_input_X, forecast_site_input_Y, shape = combine_reservoir_driver_data(
                                                            forecast,
                                                            forecast_reservoir,
                                                            site,
                                                            target_variable="max_temp_c" # We know the dwallin_temp_c is the target variable
                                                                                             # from using ".head()"
                                                            )
#may be dealing with some missing values in the reservoir, so want to interpolate those
new_forecast_site_input_X = pd.DataFrame(forecast_site_input_X.copy(), columns=forecast_site_input_X.columns)
new_forecast_site_input_X.interpolate(inplace=True)
new_forecast_site_input_X.isna().sum()
print(new_forecast_site_input_X.head())
#now we want to normalize the values --> this uses the same scaler from the finetune train section to normalize the forecast values.
forecast_site_input_X = x_scaler.transform(forecast_site_input_X)
forecast_site_input_Y = y_scaler.transform(forecast_site_input_Y.values.reshape(-1, 1))

#-------- EXPORT------#
#export the pretraining data:
#NOTE: CHANGE THE PATH OF EXPORT BASED ON WHERE YOU'RE USING THIS FILE----X
pd.DataFrame(pretrain_site_input_X_train, columns=pretrain_site_input_X.columns).to_csv(f"{data_f}/pretrain_site_input_X_train.csv", index=False)
pd.DataFrame(pretrain_site_input_X_test, columns=pretrain_site_input_X.columns).to_csv(f"{data_f}/pretrain_site_input_X_test.csv", index=False)
pd.DataFrame(pretrain_site_input_Y_train, columns=["dwallin_temp_c"]).to_csv(f"{data_f}/pretrain_site_input_Y_train.csv", index=False)
pd.DataFrame(pretrain_site_input_Y_test, columns=["dwallin_temp_c"]).to_csv(f"{data_f}/pretrain_site_input_Y_test.csv", index=False)
#export the finetuning data:
pd.DataFrame(finetune_site_input_X_train, columns=finetune_site_input_X.columns).to_csv(f"{data_f}/finetune_site_input_X_train.csv", index=False)
pd.DataFrame(finetune_site_input_X_test, columns=finetune_site_input_X.columns).to_csv(f"{data_f}/finetune_site_input_X_test.csv", index=False)
pd.DataFrame(finetune_site_input_Y_train, columns=["mean_temp_c"]).to_csv(f"{data_f}/finetune_site_input_Y_train.csv", index=False)
pd.DataFrame(finetune_site_input_Y_test, columns=["mean_temp_c"]).to_csv(f"{data_f}/finetune_site_input_Y_test.csv", index=False)
#export the forecasting data:
pd.DataFrame(forecast_site_input_X, columns=new_forecast_site_input_X.columns).to_csv(f"{data_f}/forecast_site_input_X.csv", index=False)
pd.DataFrame(forecast_site_input_Y, columns=["max_temp_c"]).to_csv(f"{data_f}/forecast_site_input_Y.csv", index=False)