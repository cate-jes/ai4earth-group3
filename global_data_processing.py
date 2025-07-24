import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



import os
path = #...
dir_list = os.listdir(path)
#print("Files and directories in '", path, "' :")
# prints all files
#print(dir_list)

data_f = #....

gridMET = pd.read_csv(f"{data_f}/gridMET_area_weighted.csv")
reservoir_release = pd.read_csv(f"{data_f}/reservoir_releases_total.csv")
dwallin_temp = pd.read_csv(f"{data_f}/dwallin_stream_preds.csv")
target_temp = pd.read_csv(f"{data_f}/temperature_observations_forecast_sites.csv")

sites = {
    1573: ["Cannonsville", "Pepacton"],
    1571: ["Cannonsville"],
    1565: ["Cannonsville"],
    1450: ["Pepacton"],
    1641: ["Neversink"]
    }
reservoirs = ["Cannonsville", "Pepacton", "Neversink"]

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

#-----Fill in our missing data, set timeframe, merge drivers + target feature-----#
test_target_temp = target_temp_.copy()
test_dwallin = dwallin_temp.copy()
target_temp_filled = pd.merge(target_temp_, dwallin_temp, on=["date", "seg_id_nat"], how="left")
#now that we've merged, replace missing values in mean_temp_c
target_temp_filled["mean_temp_c"] = np.where(target_temp_filled["mean_temp_c"].isna(), target_temp_filled["dwallin_temp_c"], target_temp_filled["mean_temp_c"])
gridMET_finetune = gridMET[(gridMET['date'] >= finetuning_time["start"]) & (gridMET['date'] <= finetuning_time["end"])]
target_temp_filled = target_temp_filled[(target_temp_filled['date'] >= finetuning_time["start"]) & (target_temp_filled['date'] <= finetuning_time["end"])]

input_data_finetune = pd.merge(gridMET_finetune, target_temp_filled, on=["seg_id_nat", "date"], how="right")

#--------PROCESSING FOR INPUT DATA -------#
ohe = OneHotEncoder()
ohe.fit(input_data_finetune[["seg_id_nat"]])

ohe_df = pd.DataFrame(ohe.transform(input_data_finetune[["seg_id_nat"]]).toarray(), columns=ohe.get_feature_names_out())
input_data_finetune = pd.concat([input_data_finetune, ohe_df], axis=1)

#COMBINE DATA --> ADD EACH RESERVOIR AS OWN COLUMN, ADD RELEASE VALUES WHERE MATCHED
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

#combine global data!
finetune_input_X, finetune_input_Y, shape = combine_global_data(
                                                            input_data_finetune,
                                                            reservoir_release,
                                                            target_variable="mean_temp_c" # We know the dwallin_temp_c is the target variable
                                                                                             # from using ".head()"
                                                            )

#TRAIN / TEST SPLITS AND Z SCORE STANDARDIZATION
#function for scaling: scale our data function. Takes in the scaler we set based on the finetune test data, and scales the data
def Scale(input_X, input_Y, x_scale, y_scale): #input is a dataframe and a series.
    #transform our data
    new_input_X = x_scaler.transform(input_X)
    new_input_Y = y_scaler.transform(input_Y.reshape(-1, 1))
    
    return new_input_X, new_input_Y

#-------Data processing -- Split and Scale based on actual input data! -------#
x_scaler = StandardScaler()
y_scaler = StandardScaler()
input_X_train, input_X_test, input_Y_train, input_Y_test = train_test_split(finetune_input_X, finetune_input_Y, test_size=0.2, shuffle=False)
#site_input_X_train/test is an dataframe, site_input_Y_train/test is a series. Must convert to numpy array to be used with scalar.fit
#convert y series to an array
input_Y_train = np.array(input_Y_train)
input_Y_test = np.array(input_Y_test)
x_scaler.fit(input_X_train)
y_scaler.fit(input_Y_train.reshape(-1, 1))
#apply scale function
final_input_X_train, final_input_Y_train = Scale(input_X_train, input_Y_train, x_scaler, y_scaler);
final_input_X_test, final_input_Y_test = Scale(input_X_test, input_Y_test, x_scaler, y_scaler);
#export training data ----->
#export the finetuning data:
pd.DataFrame(final_input_X_train, columns=input_X_train.columns).to_csv(f"{data_f}/finetune_global_X_train.csv", index=False)
pd.DataFrame(final_input_X_test, columns=input_X_test.columns).to_csv(f"{data_f}/finetune_global_input_X_test.csv", index=False)
pd.DataFrame(final_input_Y_train, columns=["mean_temp_c"]).to_csv(f"{data_f}/finetune_global_Y_train.csv", index=False)
pd.DataFrame(final_input_Y_test, columns=["mean_temp_c"]).to_csv(f"{data_f}/finetune_global_Y_test.csv", index=False)

#--------PROCESSING FORECAST DATA----------#
forecast = pd.read_csv(f"{data_f}/forecast_data_E0.csv")
forecast_reservoir = pd.read_csv(f"{data_f}/forecast_release_data.csv")
#replacing missing values and dropping unlabeled sites,
forecast["max_temp_c"] = forecast["max_temp_c"].interpolate()
forecast = forecast.dropna(subset=["site_id"]).copy()
#converting both to correct date time format
forecast["date"] = pd.to_datetime(forecast["date"])
forecast_reservoir['date'] = pd.to_datetime(forecast_reservoir['date'])
#add a one hot encoding
ohe_df = pd.DataFrame(ohe.transform(forecast[["seg_id_nat"]]).toarray(), columns=ohe.get_feature_names_out())
forecast = pd.concat([forecast, ohe_df], axis=1)
#combine the forecast date with the forecast reservoir data --> For testing the model.
forecast_input_X, forecast_input_Y, shape = combine_global_data(
                                                            forecast,
                                                            forecast_reservoir,
                                                            target_variable="max_temp_c" # We know the dwallin_temp_c is the target variable
                                                                                             # from using ".head()"
                                                            )
#deal with missing values
new_forecast_input_X = pd.DataFrame(forecast_input_X.copy(), columns=forecast_input_X.columns)
new_forecast_input_X.interpolate(inplace=True)
#convert to array for scaling
new_forecast_Y = np.array(forecast_input_Y)
#now we want to normalize the values --> this uses the same scaler from the finetune train section to normalize the forecast values.
final_forecast_input_X, final_forecast_input_Y = Scale(forecast_input_X, new_forecast_Y, x_scaler, y_scaler)
#export data
pd.DataFrame(final_forecast_input_X, columns=forecast_input_X.columns).to_csv(f"{data_f}/forecast_global_input_X.csv", index=False)
pd.DataFrame(final_forecast_input_Y, columns=["max_temp_c"]).to_csv(f"{data_f}/forecast_global_input_Y.csv", index=False)


