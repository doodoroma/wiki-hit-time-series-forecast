import numpy as np 
import pandas as pd
import datetime as dt
import itertools
from supersmoother import SuperSmoother
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

from main.utils.utils import smape

def clean(series, limit_range = 5, seasonality_th = 0.6):
  n_series = len(series)
  stl = STL(series, period = 7, robust = True)
  res = stl.fit()
  detrend = series - res.trend 

  if (1 - np.var(res.resid) / np.var(detrend)) >= seasonality_th:
    series = res.trend + res.resid # deseasonlized series

  tt = np.arange(n_series)
  model = SuperSmoother()
  model.fit(tt, series)
  yfit = model.predict(tt)
  resid = series - yfit

  resid_q = np.quantile(resid, [0.25, 0.75])
  iqr = np.diff(resid_q)
  limits = resid_q + limit_range * iqr * [-1, 1]

  outliers = (limits[0] > resid) | (resid > limits[1])
  cleaned = series.copy()
  cleaned[outliers] = cleaned.rolling(window=5, min_periods=1, center=True).mean()[outliers]
  return cleaned

def plot_all_methods(series_train_and_valid, series_train, HORIZON):
    [
        plot_eval(
            y_true=series_train_and_valid[series_name],
            y_pred=naive_forecast(
                series_train, 
                series_name=series_name, 
                method=last_value, 
                HORIZON=HORIZON
            ),
            method_name = 'Last Value',
            accuracy_measure=smape
        ) for series_name in itertools.islice(series_train.columns, 27, 28)
    ]
    [
        plot_eval(
            y_true=series_train_and_valid[series_name],
            y_pred=naive_forecast(
                series_train, 
                series_name=series_name, 
                method=last_average,
                history=7, 
                HORIZON=HORIZON
            ),
            method_name = 'last average',
            accuracy_measure=smape
        ) for series_name in itertools.islice(series_train.columns, 27, 28)
    ]

    [
        plot_eval(
            y_true=series_train_and_valid[series_name],
            y_pred=naive_forecast(
                series_train, 
                series_name=series_name, 
                method=moving_average,
                history=7, 
                HORIZON=HORIZON
            ),
            method_name = 'moving average',
            accuracy_measure=smape
        ) for series_name in itertools.islice(series_train.columns, 27, 28)
    ]

    [
        plot_eval(
            y_true=series_train_and_valid[series_name],
            y_pred=naive_forecast(
                series_train, 
                series_name=series_name, 
                method=last_season,
                period=7, 
                HORIZON=HORIZON
            ),
            method_name = 'last season',
            accuracy_measure=smape
        ) for series_name in itertools.islice(series_train.columns, 27, 28)
    ]

    [
        plot_eval(
            y_true=series_train_and_valid[series_name],
            y_pred=naive_forecast(
                series_train, 
                series_name=series_name, 
                method=last_average_season,
                period=7, 
                history=3,
                HORIZON=HORIZON
            ),
            method_name = 'last average season',
            accuracy_measure=smape
        ) for series_name in itertools.islice(series_train.columns, 27, 28)
    ]

    [
        plot_eval(
            y_true=series_train_and_valid[series_name],
            y_pred=naive_forecast(
                series_train, 
                series_name=series_name, 
                method=moving_average_season,
                period=7, 
                history=3,
                HORIZON=HORIZON
            ),
            method_name = 'moving average season',
            accuracy_measure=smape
        ) for series_name in itertools.islice(series_train.columns, 27, 28)
    ]

    [
        plot_eval(
            y_true=series_train_and_valid[series_name],
            y_pred=naive_forecast(
                series_train, 
                series_name=series_name, 
                method=drift_method,
                trend_history=21,
                HORIZON=HORIZON
            ),
            method_name = 'drift',
            accuracy_measure=smape
        ) for series_name in itertools.islice(series_train.columns, 27, 28)
    ]

    [
        plot_eval(
            y_true=series_train_and_valid[series_name],
            y_pred=naive_forecast(
                series_train, 
                series_name=series_name, 
                method=decomposite_forecast,
                trend_history = 32,
                HORIZON=HORIZON
            ),
            method_name = 'decomposite_forecast',
            accuracy_measure=smape
        ) for series_name in itertools.islice(series_train.columns, 27, 28)
    ]

def plot_eval(y_true, y_pred, accuracy_measure, method_name = 'Naive', with_residual_diagnostic = False):
    y_valid = y_true.loc[y_pred.index[0]:]
    y_history = y_true.loc[:y_pred.index[0]]


    plt.plot(y_history, 'b--')
    plt.plot(y_valid, 'bo--')
    plt.plot(y_pred, 'go-')
    acc = accuracy_measure(y_true, y_pred[:len(y_true)])
    plt.title(f'"{method_name}" method: {y_true.name} - {accuracy_measure.__name__}={acc:.2f}')
    plt.legend(['true history', 'truth', 'prediction'])
    plt.show()
    
    if with_residual_diagnostic:
        e_t = y_valid - y_pred
        plt.plot(e_t, 'ro-')
        plt.title('Residual diagnostic')
        plt.show()

        e_t_mean = e_t.mean()
        plot_acf(e_t, title=f'residual autocorrelation. e_t_mean = {e_t_mean}')
    return y_true.name, acc

def eval_accuracy(y_true, y_pred, accuracy_measure):
    return accuracy_measure(y_true, y_pred[:len(y_true)])

def last_value(train, HORIZON):
    return [train[-1]] * HORIZON

def last_average(train, HORIZON, history):
    return [train.tail(history).mean()] * HORIZON

def moving_average(train, HORIZON, history):
    pred = []
    for i in range(HORIZON):
        pred.append(np.append(train.tail(history - i).values, pred).mean())
    return pred

def last_season(train, HORIZON, period):
    assert HORIZON % period == 0
    return pd.concat([train[-period:]] * int(HORIZON / period)).values

def last_average_season(train, HORIZON, period, history):
    assert HORIZON % period == 0
    season_avg = np.mean(np.stack([train[-period*(h+1):len(train)-period*h] for h in range(history)]), axis=0)
    return np.repeat(season_avg, int(HORIZON / period))

def moving_average_season(train, HORIZON, period, history):
    pred = []
    for i in range(int(HORIZON / period)):
        train_with_pred = np.append(train, pred)
        pred.append(np.mean(np.stack([train_with_pred[-period*(h+1):len(train_with_pred)-period*h] for h in range(history)]), axis=0))
    return np.hstack(pred)

def drift_method(train, HORIZON, trend_history = None):
    if trend_history is None:
        T = len(train)
    else:
        T = trend_history
    y_1 = train[-T]
    y_T = train[-1]
    pred = []
    for _ in range(HORIZON):
        y_T_h = y_T + 1 * ((y_T - y_1) / (T - 1))
        pred.append(y_T_h)
        y_T = y_T_h
        T = T + 1
    return pred


def decomposite_forecast(train, HORIZON, period = 7, trend_history = None):
    stl = STL(train, period = period, robust = True)
    res = stl.fit()
    y_pred_trend = drift_method(res.trend, HORIZON, trend_history)
    y_pred_season = last_season(res.seasonal, HORIZON, period=period)
    return y_pred_trend + y_pred_season

def naive_forecast(
    train,
    series_name = 'series-1',
    method = last_value,
    **kwargs
):
    HORIZON = kwargs.get('HORIZON', 21)
    dates = pd.date_range(
        train.index[-1] + dt.timedelta(days=1),
        train.index[-1] + dt.timedelta(days=HORIZON)
    )
    forecast_values = method(train[series_name], **kwargs)
    return pd.Series(
        forecast_values,
        dates
    )