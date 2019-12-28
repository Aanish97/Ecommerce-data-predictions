import pandas as pd

#reading csv into dataframe
#"D:/.Semester 7/FYP/FYP docx/data.csv"
df = pd.read_csv("D:/.Semester 7/FYP/FYP docx/data.csv", encoding="ISO-8859-1")

#de_duplication
df.drop_duplicates(subset=['InvoiceNo', 'Description', 'Quantity', 'InvoiceDate', 'CustomerID', 'UnitPrice', 'Country'], keep='first', inplace=True)

#date_format_standardization
if '/' in df['InvoiceDate']:
    df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
    df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%e -%m -%Y')
else:
    df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
    df['InvoiceDate'] = df['InvoiceDate'].dt.strftime('%m -%e -%Y')


#transaction wise data
new_df = df.groupby(['InvoiceNo', 'InvoiceDate', 'CustomerID', 'Country'])['UnitPrice'].sum()
new_dff = new_df.to_frame()

#export_csv = new_dff.to_csv (r'D:\.Semester 7\FYP\FYP docx\data_updated.csv', header=True)
#new_dff

new_dff.to_excel('after_ETL.xlsx')

print(df.head)

































