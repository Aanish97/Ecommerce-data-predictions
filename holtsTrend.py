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
data = pd.Series(df_train["UnitPrice"].values, df_train["InvoiceDate"])
print(data)

#ax=ans.plot()
#ax.set_xlabel("Date")
#ax.set_ylabel("Unit Price")
#plt.show()


#exponential linear trend

#fit1 = SimpleExpSmoothing(data).fit(smoothing_level=0.2,optimized=False)
#fcast1 = fit1.forecast(3).rename(r'$\alpha=0.2$')
#fit2 = SimpleExpSmoothing(data).fit(smoothing_level=0.6,optimized=False)
#fcast2 = fit2.forecast(3).rename(r'$\alpha=0.6$')
#fit3 = SimpleExpSmoothing(data).fit()
#fcast3 = fit3.forecast(3).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])

#ax = data.plot(marker='o', color='black', figsize=(12,8))
#fcast1.plot(marker='o', ax=ax, color='blue', legend=True)
#fit1.fittedvalues.plot(marker='o', ax=ax, color='blue')
#fcast2.plot(marker='o', ax=ax, color='red', legend=True)

#fit2.fittedvalues.plot(marker='o', ax=ax, color='red')
#fcast3.plot(marker='o', ax=ax, color='green', legend=True)
#fit3.fittedvalues.plot(marker='o', ax=ax, color='green')
#plt.show()


#holts linear trend
fit1 = Holt(data).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast1 = fit1.forecast(5).rename("Holt's linear trend")

#fit2 = Holt(data, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
#fcast2 = fit2.forecast(5).rename("Exponential trend")
#fit3 = Holt(data, damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
#fcast3 = fit3.forecast(5).rename("Additive damped trend")

ax = data.plot(color="black", marker="o", figsize=(12,8))

fit1.fittedvalues.plot(ax=ax, color='blue')
fcast1.plot(ax=ax, color='blue', marker="o", legend=True)

#fit2.fittedvalues.plot(ax=ax, color='red')
#fcast2.plot(ax=ax, color='red', marker="o", legend=True)
#fit3.fittedvalues.plot(ax=ax, color='green')
#fcast3.plot(ax=ax, color='green', marker="o", legend=True)

plt.show()

