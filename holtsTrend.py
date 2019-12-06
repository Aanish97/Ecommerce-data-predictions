import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.model_selection import train_test_split


df = pd.read_excel("after_ETL.xlsx", encoding="ISO-8859-1", sep=';')

df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%Y-%m-%d')


df = df.groupby(['InvoiceDate'])['UnitPrice'].sum()
df = df.to_frame()

df.to_excel('holtsTrend.xlsx')
df = pd.read_excel("holtsTrend.xlsx", encoding="ISO-8859-1", sep=';')


start_count = int(len(df)*0.75)

df_train=df[0:start_count]
df_test=df[start_count+1:]

#print(df_train["UnitPrice"])
#print(df_train["InvoiceDate"])
index= pd.date_range(start='01-2010', end='12-2011', freq='M')
data_train = pd.Series(df_train["UnitPrice"].values, df_train["InvoiceDate"])
data_test = pd.Series(df_test["UnitPrice"].values, df_test["InvoiceDate"])
print(data_train)



#holts linear trend
fit1 = Holt(data_train).fit(smoothing_level=0.6, smoothing_slope=0.4, optimized=False)
fcast1 = fit1.forecast(30).rename("Holt's linear trend")
#fit2 = Holt(data, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
#fcast2 = fit2.forecast(5).rename("Exponential trend")
#additive damped trend
#fit3 = Holt(data_train, damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
#fcast3 = fit3.forecast(50).rename("Additive damped trend")


ax = df.plot(color="black", marker="o", figsize=(12,8))
print("---------------------")
print(fit1.fittedvalues)    #holts trend values for training data

#ax = data_test.plot(color="yellow", marker="o", figsize=(12,8))

fit1.fittedvalues.plot(ax=ax, color='blue')
fcast1.plot(ax=ax, color='blue', marker="o", legend=True)
#fit2.fittedvalues.plot(ax=ax, color='red')
#fcast2.plot(ax=ax, color='red', marker="o", legend=True)
#fit3.fittedvalues.plot(ax=ax, color='green')
#fcast3.plot(ax=ax, color='green', marker="o", legend=True)

plt.show()

