import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as Model
from sklearn.metrics import mean_absolute_percentage_error
df = pd.read_csv('house_price_regression_dataset.csv')

df.info()

X = df.drop('House_Price',axis=1)
y = df['House_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = Model()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(mean_absolute_percentage_error(y_test,y_pred))

from sklearn.decomposition import PCA
import  matplotlib.pyplot as plt

pca = PCA(n_components=1)

pca_data = pca.fit_transform(X_test)
# print(pca_data)
plt.scatter(x=pca_data,y=y_test)
plt.scatter(x=pca_data,y=y_pred,color='red')
plt.show()