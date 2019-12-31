import pyfpgrowth
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
patterns = pyfpgrowth.find_frequent_patterns(tid, 20)
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)

print(rules)
print("these are rules")









