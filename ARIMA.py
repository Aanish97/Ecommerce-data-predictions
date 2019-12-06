from pandas import read_csv
import pandas as pd
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
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


X = df.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=False, transparams=False, methods='nm', maxiter=200)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()