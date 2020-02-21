from __future__ import division
from datetime import datetime, timedelta,date
from sklearn.cluster import KMeans
import pandas as pd

#reading data.csv
data = pd.read_csv("D:\.Semester 8\FYP 1\FYP docx/data.csv", encoding="ISO-8859-1")
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

#creating for 6months & 3months
last6Months = data[(data.InvoiceDate >= date(2011,6,1)) & (data.InvoiceDate < date(2011,12,1))].reset_index(drop=True)
last3Months = data[(data.InvoiceDate < date(2011,6,1)) & (data.InvoiceDate >= date(2011,3,1))].reset_index(drop=True)

#create users for assigning clustering
users = pd.DataFrame(last6Months['CustomerID'].unique())
users.columns = ['CustomerID']

#order cluster method
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

#recency scoring
print('recency calculating...')
tx_max_purchase = last6Months.groupby('CustomerID').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
users = pd.merge(users, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(users[['Recency']])
users['RecencyCluster'] = kmeans.predict(users[['Recency']])

users = order_cluster('RecencyCluster', 'Recency',users,False)

#frequency scoring
print('frequency calculating...')
tx_frequency = last6Months.groupby('CustomerID').InvoiceDate.count().reset_index()
tx_frequency.columns = ['CustomerID','Frequency']
users = pd.merge(users, tx_frequency, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(users[['Frequency']])
users['FrequencyCluster'] = kmeans.predict(users[['Frequency']])

users = order_cluster('FrequencyCluster', 'Frequency',users,True)

#revenue scoring
print('revenue calculating...')
last6Months['Revenue'] = last6Months['UnitPrice'] * last6Months['Quantity']
tx_revenue = last6Months.groupby('CustomerID').Revenue.sum().reset_index()
users = pd.merge(users, tx_revenue, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(users[['Revenue']])
users['RevenueCluster'] = kmeans.predict(users[['Revenue']])
users = order_cluster('RevenueCluster', 'Revenue',users,True)

#overall scoring
users['OverallScore'] = users['RecencyCluster'] + users['FrequencyCluster'] + users['RevenueCluster']
users['Segment'] = 'Low-Value'
users.loc[users['OverallScore']>2,'Segment'] = 'Mid-Value'
users.loc[users['OverallScore']>4,'Segment'] = 'High-Value'

print(users.head)

last3Months['Revenue'] = last3Months['UnitPrice'] * last3Months['Quantity']
users3Months = last3Months.groupby('CustomerID')['Revenue'].sum().reset_index()
users3Months.columns = ['CustomerID','m3_Revenue']

tx_merge = pd.merge(users, last6Months, on='CustomerID', how='left')
tx_merge = tx_merge.fillna(0)

tx_graph = tx_merge.query("m6_Revenue < 30000")

#RFM completed, continuing LTV scores

import xgboost as xgb
#from sklearn.model_selection import KFold, cross_val_score, train_test_split


import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

plot_data = [
    go.Histogram(
        x=users3Months.query('m3_Revenue < 10000')['m3_Revenue']
    )
]

plot_layout = go.Layout(
        title='3 month Revenue'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
