import pandas as pd

data = pd.read_csv('weather.csv')

print ('Data Shape:', data.shape)
print ('Data Columns:', data.columns)

X = data[data.columns[:-1]]
Y = data[data.columns[-1]]

print ('Feature List', X.head())
print ('Response', Y.head())