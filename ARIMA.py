from pandas import read_csv
import numpy as np
from datetime import datetime
import pandas as pd
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


df = pd.read_excel("after_ETL.xlsx", encoding="ISO-8859-1", sep=';')

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%Y-%m-%d')


df = df.groupby(['InvoiceDate'])['UnitPrice'].sum()
df = df.to_frame()

df.to_excel('ARIMA.xlsx')
df = pd.read_excel("ARIMA.xlsx", encoding="ISO-8859-1", sep=';')

df = pd.Series(df["UnitPrice"].values, df["InvoiceDate"])

X = df.values#np.log(df.values)
size = int(len(X) * .75)
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(3, 1, 1))#order of autoregressive model, order of degree of difference, order of moving average
    model_fit = model.fit(disp=False, start_ar_lags=2 * max(3, 1), transparams=False, methods='nm', maxiter=200)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %f' % error)
pyplot.plot(test)

actual_forecast=list()
for t in range(50):#how much into future
    model = ARIMA(history, order=(3, 1, 1))#order of autoregressive model, order of degree of difference, order of moving average
    model_fit = model.fit(disp=False, start_ar_lags=2 * max(3, 1), transparams=False, methods='nm', maxiter=200)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    actual_forecast.append(yhat)
    history.append(yhat)
    print(yhat)

pyplot.plot(predictions, color='red')
pyplot.show()


df.to_excel('ARIMA.xlsx')
df = pd.read_excel("ARIMA.xlsx", encoding="ISO-8859-1", sep=';')
date = df['InvoiceDate'][len(df)-1]
print(date)

new_dff = pd.DataFrame(actual_forecast)

from datetime import datetime as dt, timedelta as td
predictions_date_list = list()
for i in range(len(actual_forecast)):
    date= (dt.strptime(str(date), '%Y-%m-%d') + td(days=1)).strftime('%Y-%m-%d')
    predictions_date_list.append(date)

#adding new column
new_dff['InvoiceDate']=predictions_date_list
#adding new predicted rows to the existing data
df.append(new_dff, sort = False)

print(type(new_dff))
print(df)

#df = df.to_frame()
df.to_excel('ARIMA.xlsx')
new_dff.to_excel('ARIMA.xlsx')