import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.offline as pyoff
import plotly.graph_objs as go
import xgboost as xgb

#read excel file in df
df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")

#we will be using only UK data for fast  running
df = df.query("Country=='United Kingdom'").reset_index(drop=True)


#----------------------------------Calculate Recency-------------------------------------

#change format of date time according to python scripts like year-month-day hour-minute-seconds
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

#print(df['InvoiceDate'])

#make a new dataframe of customers unique ids
user_unique_id = pd.DataFrame(df['CustomerID'].unique())
user_unique_id.columns = ['CustomerID']

#print(customer_id['CustomerID'])

#get date of recent purchase of every customer and make new dafaframe with customer id plus recent purchase date
recent_purchase = df.groupby('CustomerID').InvoiceDate.max().reset_index()
recent_purchase.columns = ['CustomerID', 'RecentPurchaseDate']

#print(recent_purchase['RecentPurchaseDate'].max())
#print(recent_purchase['RecentPurchaseDate'])
#print((recent_purchase['RecentPurchaseDate'].max()-recent_purchase['RecentPurchaseDate']).dt.days)

#make a new column in recent_purchase days(score)
recent_purchase['Recency'] = (recent_purchase['RecentPurchaseDate'].max() - recent_purchase['RecentPurchaseDate']).dt.days

#attach these scores to customer unique ids
user_unique_id = pd.merge(user_unique_id, recent_purchase[['CustomerID', 'Recency']], on='CustomerID')

#print(user_unique_id)

#retunr top 5 rows
#print(user_unique_id.head())

#print(user_unique_id)

plot_data = [
    go.Histogram(
        x=user_unique_id['Recency']
    )
]

plot_layout = go.Layout(
        title='Histogram: Frequency(On Vertical Axis) vs Recency Score(On Horizontal Axis)'
)
fig = go.Figure(data=plot_data, layout=plot_layout)
#pyoff.plot(fig)

#print(sum(user_unique_id.Recency))

#print(user_unique_id.head())

#print(user_unique_id.Recency.describe())


#how many clusters are required
sse = {}
tx_recency = user_unique_id[['Recency']]
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
    #tx_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("Within cluster sum of squares distances")
#plt.show()

#build 4 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=4)
kmeans.fit(user_unique_id[['Recency']])
user_unique_id['RecencyCluster'] = kmeans.predict(user_unique_id[['Recency']])

#print(user_unique_id['RecencyCluster'].describe())

#function for ordering cluster numbers


#print(user_unique_id.groupby('RecencyCluster')['Recency'].describe())


def order_cluster(cluster_field_name, target_field_name, df, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df, df_new[[cluster_field_name, 'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name], axis=1)
    df_final = df_final.rename(columns={"index": cluster_field_name})
    return df_final


user_unique_id = order_cluster('RecencyCluster', 'Recency', user_unique_id, False)

#print(user_unique_id.groupby('RecencyCluster')['Recency'].describe())

#---------------------------------Frequency--------------------------------------------

user_frequency = df.groupby('CustomerID').InvoiceDate.count().reset_index()
user_frequency.columns = ['CustomerID', 'Frequency']

#add this data to our main dataframe
user_unique_id = pd.merge(user_unique_id, user_frequency, on='CustomerID')

#plot the histogram
plot_data = [
    go.Histogram(
        x=user_unique_id.query('Frequency < 1000')['Frequency']
    )
]

plot_layout = go.Layout(
        title='Histogram: Frequency(On Vertical Axis) vs Frequency Score(On Horizontal Axis)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
#pyoff.plot(fig)

#print(user_unique_id)

#k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(user_unique_id[['Frequency']])
user_unique_id['FrequencyCluster'] = kmeans.predict(user_unique_id[['Frequency']])

#order the frequency cluster
tx_user = order_cluster('FrequencyCluster', 'Frequency', user_unique_id, True)

#see details of each cluster
#print(tx_user.groupby('FrequencyCluster')['Frequency'].describe())

#---------------------Monetary-----------------------------------------------------------

#calculate revenue for each customer
df['Revenue'] = df['UnitPrice'] * df['Quantity']
user_revenue = df.groupby('CustomerID').Revenue.sum().reset_index()

#merge it with our main dataframe
user_unique_id = pd.merge(user_unique_id, user_revenue, on='CustomerID')

#plot the histogram
plot_data = [
    go.Histogram(
        x=user_unique_id.query('Revenue < 10000')['Revenue']
    )
]

plot_layout = go.Layout(
        title='Histogram: Frequency(On Vertical Axis) vs Monetary Score(On Horizontal Axis)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
#pyoff.plot(fig)

#apply clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(user_unique_id[['Revenue']])
user_unique_id['RevenueCluster'] = kmeans.predict(user_unique_id[['Revenue']])


#order the cluster numbers
user_unique_id = order_cluster('RevenueCluster', 'Revenue', user_unique_id, True)

#show details of the dataframe
#print(user_unique_id.groupby('RevenueCluster')['Revenue'].describe())



#----------------------------------------Final result--------------------------------------------------------

#calculate overall score and use mean() to see details
user_unique_id['OverallScore'] = user_unique_id['RecencyCluster'] + user_unique_id['FrequencyCluster'] + user_unique_id['RevenueCluster']
#print(user_unique_id)
#print(user_unique_id.groupby('OverallScore')['Recency', 'Frequency', 'Revenue'].mean())

user_unique_id['Segment'] = 'Low-Value'
user_unique_id.loc[user_unique_id['OverallScore']>2,'Segment'] = 'Mid-Value'
user_unique_id.loc[user_unique_id['OverallScore']>4,'Segment'] = 'High-Value'

#print(user_unique_id)



#----------------------------------------LTV--------------------------------------

df['revenue'] = df['UnitPrice'] * df['Quantity']

unique_id_revenue = df.groupby('CustomerID')['revenue'].sum().reset_index()

plt.xlim(xmin=min(unique_id_revenue['revenue']), xmax=10000)
plt.hist(unique_id_revenue['revenue'], bins='auto')
#plt.show()

#print(min(unique_id_revenue.query('revenue<0')['CustomerID']))


users_merge = pd.merge(user_unique_id, unique_id_revenue, on='CustomerID', how='left')

#print(users_merge)

for_graph = users_merge.query("revenue < 30000")

plot_data = [
    go.Scatter(
        x=for_graph.query("Segment == 'Low-Value'")['OverallScore'],
        y=for_graph.query("Segment == 'Low-Value'")['revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=for_graph.query("Segment == 'Mid-Value'")['OverallScore'],
        y=for_graph.query("Segment == 'Mid-Value'")['revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=for_graph.query("Segment == 'High-Value'")['OverallScore'],
        y=for_graph.query("Segment == 'High-Value'")['revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "LTV"},
        xaxis= {'title': "RFM Score"},
        title='LTV'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
#pyoff.plot(fig)

#remove outliers
users_merge = users_merge[users_merge['revenue']<users_merge['revenue'].quantile(0.99)]


#creating 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(users_merge[['revenue']])
users_merge['LTVCluster'] = kmeans.predict(users_merge[['revenue']])

#order cluster number based on LTV
users_merge = order_cluster('LTVCluster', 'revenue', users_merge, True)

#creatinga new cluster dataframe
tx_cluster = users_merge.copy()

#see details of the clusters
print(tx_cluster.groupby('LTVCluster')['revenue'].describe())

#convert categorical columns to numerical
tx_class = pd.get_dummies(tx_cluster)

#calculate and show correlations
corr_matrix = tx_class.corr()
corr_matrix['LTVCluster'].sort_values(ascending=False)

#create X and y, X will be feature set and y is the label - LTV
X = tx_class.drop(['LTVCluster', 'revenue'], axis=1)
y = tx_class['LTVCluster']

#split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)


print(tx_cluster.head())
print(tx_class.head())


