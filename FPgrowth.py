from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt

import sys
import pandas as pd

df = pd.read_csv("D:/.Semester 7/FYP/FYP docx/data.csv", encoding="ISO-8859-1")
df.drop_duplicates(subset=['InvoiceNo', 'Description', 'Quantity', 'InvoiceDate', 'CustomerID', 'UnitPrice', 'Country'], keep='first', inplace=True)

if '/' in df['InvoiceDate']:
    df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
    df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%e -%m -%Y')
else:
    df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
    df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%m -%e -%Y')

unique_items = df['Description'].unique()
unique_items = unique_items.tolist()
print(unique_items)

temp=str(df['InvoiceNo'][0])
print(temp)

tid = list()
temp_tid = list()
for index, rows in df.iterrows():
    if temp == rows['InvoiceNo']:
        temp_tid.append(unique_items.index(rows['Description']))
    else:
        temp_tid.sort()
        tid.append(temp_tid)#append transaction to transactions list
        temp_tid = list()
        temp = str(rows['InvoiceNo'])
        temp_tid.append(unique_items.index(rows['Description']))

print(tid)

sys.setrecursionlimit(25000)

te = TransactionEncoder()
te_ary = te.fit(tid).transform(tid)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True)
print("frequent items")
print(type(frequent_itemsets))
print(frequent_itemsets)

frequent_itemsets['itemsets']= [list(x) for x in frequent_itemsets['itemsets']]
print(type(frequent_itemsets['itemsets']))

itemslist=list()
for index, rows in frequent_itemsets.iterrows():
    il=list()
    for f in range(len(rows['itemsets'])):
        il.append(unique_items[rows['itemsets'][f]])
        print(rows['itemsets'][f])
        print(unique_items[rows['itemsets'][f]])
    itemslist.append(il)

frequent_itemsets['itemsets']=itemslist
frequent_itemsets.to_excel('FPgrowth.xlsx')
