from __future__ import division
from pandas import datetime
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.offline as pyoff
import plotly.graph_objs as go
import chart_studio

def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')

df = pd.read_excel("after_ETL.xlsx", encoding="ISO-8859-1", sep=';')

df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%Y-%m-%d')

print(df.head)


tx_user = pd.DataFrame(df['CustomerID'].unique())
tx_user.columns = ['CustomerID']

#get the max purchase date for each customer and create a dataframe with it
tx_max_purchase = df.groupby('CustomerID').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']

#we take our observation point as the max invoice date in our dataset
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days

#merge this dataframe to our new user dataframe
tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')

tx_user.head()

#plot a recency histogram

plot_data = [
    go.Histogram(
        x=tx_user['Recency']
    )
]

plot_layout = go.Layout(
        title='Recency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)