import pandas as pd

# Read Excel file
df = pd.read_excel('training_data/Stock A.xlsx')

print('Data shape:', df.shape)
print('Columns:', df.columns.tolist())
print('\nFirst 10 rows:')
print(df.head(10))
print('\nLast 10 rows:')
print(df.tail(10))
print('\nData types:')
print(df.dtypes)

# Check if there is a date column
if 'Date' in df.columns:
    print('\nDate range:')
    print('Min date:', df['Date'].min())
    print('Max date:', df['Date'].max())

# Also check for lowercase 'date' column
if 'date' in df.columns:
    print('\nDate range (lowercase):')
    print('Min date:', df['date'].min())
    print('Max date:', df['date'].max())
