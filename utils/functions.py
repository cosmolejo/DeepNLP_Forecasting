import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesPredictor

def plot_timeseries_forecasting(historical_data, low, median, high, start_forecasting_date=-1,
                                prediction_interval = .8, prediction_length = 12,
                                figsize=(20,5)):

    years_data = historical_data.index
    start = pd.date_range(years_data[-1], periods=2, freq="MS")[-1] if start_forecasting_date == -1 else start_forecasting_date
    forecast_index = pd.date_range(start=start,
                                   periods=prediction_length, freq="MS")

    plt.figure(figsize=figsize)
    plt.plot(years_data, historical_data,
             color="royalblue", label="Historical data")
    plt.plot(forecast_index, median, color="tomato", label="Median forecast")
    plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3,
                     label=f"{prediction_interval * 100}% prediction interval")

    plt.axvline(x=start, color='gray', linestyle='--', linewidth=2)
    plt.legend()
    plt.grid()
    plt.show()

def fit_models(prediction_length,train_data,historical_length):
    
        pred_wql=TimeSeriesPredictor(prediction_length=prediction_length,eval_metric="WQL").fit(
            train_data.iloc[-historical_length:], presets="bolt_small" #if use_cuda else "bolt_small"
            ,
        )

        pred_mase=TimeSeriesPredictor(prediction_length=prediction_length,eval_metric="MASE").fit(
            train_data.iloc[-historical_length:], presets="bolt_small" #if use_cuda else "bolt_small"
            ,
        )

        return pred_wql, pred_mase

def update_table(errors_df, row, len_train,percentege_train, len_test,percentege_test, MASE, WQL): 

    if row!=0: 
        errors_df.loc[row, "Diference_MASE"] = np.abs(MASE - errors_df.loc[row-1, "MASE_error"])
        errors_df.loc[row, "Diference_WQL"] = np.abs(WQL - errors_df.loc[row-1, "WQL_error"])

    else:
        errors_df.loc[row, "Diference_MASE"] = 0
        errors_df.loc[row, "Diference_WQL"] = 0
    
    errors_df.loc[row, "Train_size"] = len_train
    errors_df.loc[row, "Percentage_train"] = percentege_train

    errors_df.loc[row, "Test_size"] = len_test
    errors_df.loc[row, "Percentage_test"] = percentege_test


    errors_df.loc[row, "MASE_error"] = MASE #The error
    errors_df.loc[row, "WQL_error"] = WQL #The initial error is with the whole dataset

    return errors_df
