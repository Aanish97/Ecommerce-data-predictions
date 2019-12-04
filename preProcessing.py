import pandas as pd

#de_duplication
df = pd.read_csv("data.csv", encoding="ISO-8859-1")
df.drop_duplicates(subset=['InvoiceNo', 'Description', 'Quantity', 'InvoiceDate', 'CustomerID', 'UnitPrice', 'Country'], keep='first', inplace=True)

#date_format_standardization
df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%e %m,%Y')


df.to_excel('after_ETL.xlsx')

print(df.head)

































