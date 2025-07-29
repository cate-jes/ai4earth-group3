
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import os
path = './data'
dir_list = os.listdir(path)
#print("Files and directories in '", path, "' :")
# prints all files
#print(dir_list)

data_f = path

gridMET = pd.read_csv(f"{data_f}/gridMET_area_weighted.csv")

# This file contains information of amount of water releases in a type of reservoir. We will need to combine this with GridMET to obtain
# our final input drivers.
reservoir_release = pd.read_csv(f"{data_f}/reservoir_releases_total.csv")

# This file contains simulated stream water temperatures obtained through a process-based model. In the paper these are used as outputs
# for model pretraining. Here, we will use these values to fill in any missing values from the real-world ground truth data.
dwallin_temp = pd.read_csv(f"{data_f}/dwallin_stream_preds.csv")

# The real-world stream temperature values are contained
target_temp = pd.read_csv(f"{data_f}/temperature_observations_forecast_sites.csv")

forecast = pd.read_csv(f"{data_f}/forecast_data_E0.csv")

forecast_reservoir = pd.read_csv(f"{data_f}/forecast_release_data.csv")
#site breakdown
sites = {
    1573: ["Cannonsville", "Pepacton"],
    1571: ["Cannonsville"],
    1565: ["Cannonsville"],
    1450: ["Pepacton"],
    1641: ["Neversink"]
    }


reservoirs = ["Cannonsville", "Pepacton", "Neversink"]

# Additionally, we are provided with the time split for model training and forecast
pretraining_time = { "start": '1985-05-01', "end": '2020-04-01'}
finetuning_time = { "start": '1985-05-01', "end": '2021-04-14'}
forecasting_time = {"start": '2021-04-16', "end": '2021-09-30'}


#-----PREPROCESSING INPUT / ADDING MISSING DATA--------#
#takes in sites, datafiles and forecasting/finetuning times
def preprocess_data(gridMET, target_temp, dwallin_temp, sites, finetuning_time, forecasting_time):
  gridMET = gridMET[gridMET["seg_id_nat"].isin(sites.keys())] #narrowing down gridMET to only containing seg_id that matches
  target_temp_ = target_temp[target_temp["seg_id_nat"].isin(sites.keys())]
  dwallin_temp = dwallin_temp[dwallin_temp["seg_id_nat"].isin(sites.keys())]
  
  #datetime conversions
  gridMET = gridMET.rename(columns={"time": "date"})
  gridMET["date"] = pd.to_datetime(gridMET["date"])
  dwallin_temp["date"] = pd.to_datetime(dwallin_temp["date"])
  target_temp_["date"] = pd.to_datetime(target_temp_["date"]).dt.tz_localize(None)
  reservoir_release["date"] = pd.to_datetime(reservoir_release["date"])
  
  # Converting values from fahrenheit to celsius
  gridMET["tmin"] = (gridMET["tmin"] -32) * (5/9)
  gridMET["tmax"] = (gridMET["tmax"] - 32) * (5/9)
  
  #Fill in missing values in target temp
  test_target_temp = target_temp_.copy()
  test_dwallin = dwallin_temp.copy()
  target_temp_filled = pd.merge(test_target_temp, test_dwallin, on=["date", "seg_id_nat"], how="left")
  
  #now that we've merged, replace missing values in mean_temp_c
  target_temp_filled["mean_temp_c"] = np.where(target_temp_filled["mean_temp_c"].isna(), target_temp_filled["dwallin_temp_c"], target_temp_filled["mean_temp_c"])
  
  #Combine finetune data with target temp
  gridMET_finetune = gridMET[(gridMET['date'] >= finetuning_time["start"]) & (gridMET['date'] <= finetuning_time["end"])]
  target_temp_filled = target_temp_filled[(target_temp_filled['date'] >= finetuning_time["start"]) & (target_temp_filled['date'] <= finetuning_time["end"])]
  input_data_finetune = pd.merge(gridMET_finetune, target_temp_filled, on=["seg_id_nat", "date"], how="right")

  #Add encoding of site ids
  ohe = OneHotEncoder()
  ohe.fit(input_data_finetune[["seg_id_nat"]])

  ohe_df = pd.DataFrame(ohe.transform(input_data_finetune[["seg_id_nat"]]).toarray(), columns=ohe.get_feature_names_out())
  input_data_finetune = pd.concat([input_data_finetune, ohe_df], axis=1)

  return input_data_finetune

#prepare forecasting data for the model
def preprocess_forecast_data(input_data, forecast, forecast_reservoir, reservoir_release): 

  #replacing missing values and dropping unlabeled sites,
  forecast["max_temp_c"] = forecast["max_temp_c"].interpolate()
  
  #converting both to correct date time format
  forecast["date"] = pd.to_datetime(forecast["date"])
  forecast_reservoir['date'] = pd.to_datetime(forecast_reservoir['date'])
  
  #want to start 30 days before, use the pretraining from gridmet
  forecasting_time = { "start": '2021-03-15', "end": '2021-04-14'}
  
  #Append 30 days of gridmet data to forecast data
  gridMET_forecast = input_data.drop(columns=["prcp", "humidity", "rhmax", "rhmin", "ws", "subseg_id", "in_time_holdout", "in_space_holdout", "test", "dwallin_temp_c"]).copy()
  forecast = forecast.drop(columns=["cd", "site_name"])
  new_gridMET_forecast = gridMET_forecast[(gridMET_forecast['date'] >= forecasting_time["start"]) & (gridMET_forecast['date'] <= forecasting_time["end"])]
  # Combine gridMET_forecast and forecast (gridMET comes first)
  full_forecast = pd.concat([new_gridMET_forecast, forecast], ignore_index=True)
  #sort by date if you want to ensure time order
  full_forecast = full_forecast.sort_values(by="date").reset_index(drop=True)
  
  #Append 30 days of reservoir_release to forecast release
  first_month_release = reservoir_release[(reservoir_release['date'] >= forecasting_time["start"]) & (reservoir_release['date'] <= forecasting_time["end"])]

  full_reservoir_release = pd.concat([first_month_release, forecast_reservoir], ignore_index=True).drop_duplicates()
  full_reservoir_release = full_reservoir_release.sort_values(by="date").reset_index(drop=True)

  return full_forecast, full_reservoir_release


# COMBINE DRIVER AND RESERVOIR DATA FUNCTION ------#
def combine_global_data(input_data, reservoir_release, target_variable=None):
    # Combining with reservoir based on site type
    site_encodings = ["seg_id_nat_1450", "seg_id_nat_1565", "seg_id_nat_1571", "seg_id_nat_1573","seg_id_nat_1641"]
    res_columns = []
    for reservoir in reservoirs:
        print(reservoir)
        release_df = reservoir_release[reservoir_release["reservoir"] == reservoir][["release_volume_cms", "date"]]
        release_df = release_df.rename(columns={"release_volume_cms": f"release_volume_cms_{reservoir}"})
        input_data = input_data.merge(release_df, on="date", how="left")
        res_columns.append(f"release_volume_cms_{reservoir}")


    #prepare site data!
    site_input_X = input_data[["tmin", "tmax", "srad", "seg_id_nat_1450", "seg_id_nat_1565", "seg_id_nat_1571", "seg_id_nat_1573","seg_id_nat_1641", *res_columns]].copy()
    if target_variable:
        site_input_X["ar1"] = input_data[target_variable].shift(1)
    site_input_Y = input_data[target_variable]

    # Drop NaN due to shifting time for ar value
    site_input_X.drop(index=0, inplace=True)
    site_input_Y.drop(index=0, inplace=True)

    return site_input_X, site_input_Y, site_input_X.shape

#SCALE AND EXPORT FUNCTION-------------#
#proces data site-wise, saves the data in a npz file, returns the mean and std of the standardization data to be added to a csv file
#proces data site-wise, saves the data in a npz file, returns the mean and std of the standardization data to be added to a csv file
def process_global_data(finetune_input_data, forecast_input_data, reservoir_release, forecast_release,
                     finetune_target, forecast_target):
  finetune_input_X, finetune_input_Y, shape = combine_global_data(finetune_input_data, reservoir_release, finetune_target)
  forecast_input_X, forecast_input_Y, shape = combine_global_data(forecast_input_data, forecast_release, forecast_target)
  finetune_input_X_train, finetune_input_X_test, finetune_input_Y_train, finetune_input_Y_test = train_test_split(finetune_input_X, finetune_input_Y, test_size=0.2, shuffle=False)

  x_scaler = StandardScaler()
  y_scaler = StandardScaler()
  #convert y to a numpy array
  npy_finetune_input_Y_train = np.array(finetune_input_Y_train)
  npy_finetune_input_Y_test = np.array(finetune_input_Y_test)

  #convert forecast Y
  npy_forecast_input_Y = np.array(forecast_input_Y)

  #fit our scaler:
  x_scaler.fit(finetune_input_X_train)
  y_scaler.fit(npy_finetune_input_Y_train.reshape(-1, 1))

  #want to return our mean
  new_finetune_input_X_train = x_scaler.transform(finetune_input_X_train)
  new_finetune_input_X_test = x_scaler.transform(finetune_input_X_test)

  new_finetune_input_Y_train = y_scaler.transform(npy_finetune_input_Y_train.reshape(-1, 1))
  new_finetune_input_Y_test = y_scaler.transform(npy_finetune_input_Y_test.reshape(-1, 1))

  #forecast
  new_forecast_input_X = x_scaler.transform(forecast_input_X)
  new_forecast_input_Y = y_scaler.transform(npy_forecast_input_Y.reshape(-1, 1))
  
  #return
  #np.savez(f"{data_f}/global_input_X", finetune_train=new_finetune_input_X_train, finetune_test=new_finetune_input_X_test, forecast_X=new_forecast_input_X, x_scaled_means=x_scaler.mean_, x_scaled_stds=x_scaler.scale_ )
  #np.savez(f"{data_f}/global_input_Y", finetune_train=new_finetune_input_Y_train, finetune_test=new_finetune_input_Y_test, forecast_Y=new_forecast_input_Y, y_scaled_mean=y_scaler.mean_, y_scaled_std=y_scaler.scale_)
  return (new_finetune_input_X_train, new_finetune_input_X_test, new_finetune_input_Y_train, new_finetune_input_Y_test, 
          new_forecast_input_X, new_forecast_input_Y, x_scaler, y_scaler, shape )

def main():
    #-----------DATA PROCESSING--------#
    #process input data
    input_data_finetune = preprocess_data(gridMET, target_temp, dwallin_temp, sites, finetuning_time, forecasting_time)
    #process forecast data
    full_forecast, full_reservoir_release = preprocess_forecast_data(input_data_finetune, forecast, forecast_reservoir, reservoir_release)
    #combine data and export it.
    new_finetune_input_X_train, new_finetune_input_X_test, new_finetune_input_Y_train, new_finetune_input_Y_test, new_forecast_input_X, new_forecast_input_Y, x_scaler, y_scaler, shape = process_global_data(
        input_data_finetune, full_forecast, reservoir_release, full_reservoir_release, "mean_temp_c", "max_temp_c"
    )
    
    return new_finetune_input_X_train, new_finetune_input_X_test, new_finetune_input_Y_train, new_finetune_input_Y_test, new_forecast_input_X, new_forecast_input_Y, x_scaler, y_scaler, shape