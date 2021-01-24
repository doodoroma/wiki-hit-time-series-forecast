import pandas as pd
import datetime as dt
import pmdarima as pm

def arima_forecaset(
    train,
    series_name = 'series-1',
    **kwargs
):
    HORIZON = kwargs.get('HORIZON', 21)
    seasonal = kwargs.get('seasonal', True)
    m = kwargs.get('m', 7)
    min_p = kwargs.get('min_p', 2)
    max_p = kwargs.get('max_p', 2)
    min_q = kwargs.get('min_q', 5)
    max_q = kwargs.get('max_q', 5)
    min_P = kwargs.get('min_P', 1)
    max_P = kwargs.get('max_P', 1)
    min_Q = kwargs.get('min_Q', 2)
    max_Q = kwargs.get('max_Q', 2)

    d = kwargs.get('d', 1)
    D = kwargs.get('D', 1)

    m = kwargs.get('m', 7)
    model = pm.auto_arima(
        train[series_name], 
        
        d = d,

        start_p = min_p, 
        max_p = max_p, 
        start_q = min_q, 
        max_q = max_q, 

        seasonal = seasonal,
        m = m if seasonal else 0,
        D = D,

        start_P = min_P, 
        max_P = max_P, 
        start_Q = min_Q, 
        max_Q = max_Q, 

        random_state=42,
        n_fits=10,

        trace = False,
        suppress_warnings=True,
        error_action='ignore'

    )
    

    dates = pd.date_range(
        train.index[-1] + dt.timedelta(days=1),
        train.index[-1] + dt.timedelta(days=HORIZON)
    )

    forecast_values = model.predict(HORIZON)

    return pd.Series(
        forecast_values,
        dates
    )
    