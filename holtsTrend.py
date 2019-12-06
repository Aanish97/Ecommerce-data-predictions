import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn


df = pd.read_excel("after_ETL.xlsx", encoding="ISO-8859-1", sep=';')

df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%Y%m')


df = df.groupby(['InvoiceDate'])['UnitPrice'].sum()
df = df.to_frame()

df.to_excel('holtsTrend.xlsx')
df = pd.read_excel("holtsTrend.xlsx", encoding="ISO-8859-1", sep=';')

#plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.plot(df["InvoiceDate"], df["UnitPrice"])
plt.show()