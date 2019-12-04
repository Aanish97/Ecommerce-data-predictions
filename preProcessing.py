import pandas as pd

#reading csv into dataframe
#"D:/.Semester 7/FYP/FYP docx/data.csv"
df = pd.read_csv("data.csv", encoding="ISO-8859-1")

#de_duplication
df.drop_duplicates(subset=['InvoiceNo', 'Description', 'Quantity', 'InvoiceDate', 'CustomerID', 'UnitPrice', 'Country'], keep='first', inplace=True)

#date_format_standardization
df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%e %m,%Y')


df.to_excel('after_ETL.xlsx')

print(df.head)

































