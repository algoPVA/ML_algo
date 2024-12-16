import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('Country-data.csv')
df.info()
print(df.head())

X = df.drop('country',axis=1)

sc = StandardScaler()

X = sc.fit_transform(X)

model = KMeans(n_clusters=10)
clusters = model.fit_predict(X)
print(clusters)

df['Rang'] = clusters
df.info()
print(df.head())
df = df.sort_values('Rang')

df.to_csv('result_country.csv')