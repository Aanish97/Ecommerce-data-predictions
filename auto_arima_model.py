import pandas as pd
from pandas import datetime
from matplotlib import pyplot
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')

df = pd.read_excel("after_ETL.xlsx", encoding="ISO-8859-1", sep=';')

df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%Y-%m-%d')


df = df.groupby(['InvoiceDate'])['UnitPrice'].sum()
df = df.to_frame()

df.to_excel('ARIMA.xlsx')
df = pd.read_excel("ARIMA.xlsx", encoding="ISO-8859-1", sep=';')

df = pd.Series(df["UnitPrice"].values, df["InvoiceDate"])

print(df)


stepwise_model = auto_arima(df, start_p=1, start_q=1,
                            max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
print(stepwise_model.aic())

size = int(len(df) * .85)
train, test = df[0:size], df[size:]

# for testing the model
stepwise_model.fit(train)
future_forecast = stepwise_model.predict(n_periods=len(test))
future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])

error = mean_squared_error(test, future_forecast)
print('Test MSE: %f' % error)

pyplot.plot(df)
pyplot.plot(future_forecast, color='red')
pyplot.show()
