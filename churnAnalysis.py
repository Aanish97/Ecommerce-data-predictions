'''
requirements are that we need only numerical columns, we will not be able to process string data
'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("telcoDataset.csv")    #loading the data into df


#some ETL
#removing empty rows
for col in df.columns:
    df[col].replace(' ', np.nan, inplace=True)
    df.dropna(subset=[col], inplace=True)

df.to_csv('telcoDataset.csv', index=False)





df = pd.read_csv("telcoDataset.csv", index_col=False)


le = LabelEncoder()
dummy_columns = [] #array for multiple value columns
for col in df.columns:
    if df[col].dtype == object and col != 'customerID':
        if df[col].nunique() == 2:
            #making the dataset binary
            df[col] = le.fit_transform(df[col])
        else:
            dummy_columns.append(col)

#apply get dummies for selected columns
df_data = pd.get_dummies(data=df, columns=dummy_columns)

#feature engineering to assign binary values to the complete data
#print(df)


#print(df.head(10))
cols=[]
for col in df.columns:
    temp = col.lower()
    print(df[col].dtype)
    if df[col].nunique() == 2 or df[col].dtype == np.object:  #binary column or string column
        #print(col)
        pass

    else:

        user_unique_id = df[[col]]

        # how many clusters are required
        sse = {}
        tx = user_unique_id[[col]]
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx)
            # tx_recency["clusters"] = kmeans.labels_
            sse[k] = kmeans.inertia_

        # plt.show()

        kmeans = KMeans(n_clusters=3)
        kmeans.fit(user_unique_id[[col]])
        temp = pd.DataFrame(user_unique_id)
        df[col] = kmeans.predict(temp[[col]])

        temp_df = pd.get_dummies(df[col])


        print(str(col))

        temp_df.columns = [str(col)+'_low', str(col)+'_mid', str(col)+'_high']

        df[str(col)+'_low'] = temp_df[str(col)+'_low']
        df[str(col)+'_mid'] = temp_df[str(col)+'_mid']
        df[str(col)+'_high'] = temp_df[str(col)+'_high']

        cols.append(str(col))

df.drop(cols, axis=1, inplace=True)
print(df.describe())
df.to_csv('churnanalysis.csv', index=False)



#excluding features which has p_value>|0.05|
'''
temp_df = df

temp_df.drop(['customerID'], axis=1, inplace=True)       #customer id needs to be generalized

from scipy import stats


df_p = pd.DataFrame()  # Matrix of p-values
for x in temp_df.columns:
    for y in temp_df.columns:
        corr = stats.pearsonr(temp_df[x], temp_df[y])

        df_p.loc[x,y] = corr[1]


print(df_p)'''


#print(df)


import xgboost as xgb
from sklearn.model_selection import train_test_split

#create feature set and labels
X = df.drop(['Churn','customerID'],axis=1)
y = df.Churn
#train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)
#building the model & printing the score
xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, objective= 'binary:logistic',n_jobs=-1).fit(X_train, y_train)
print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(xgb_model.score(X_test[X_train.columns], y_test)))
