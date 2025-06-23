import matplotlib.pyplot as plt
import pandas as pd


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