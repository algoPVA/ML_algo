from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prepare_data1 import x,y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# print(x_train.iloc[0])
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# me = [[2,0,13,0,0,17,0,1,0,1]]
# me = sc.transform(me)
# print(me)
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
# predict_me = classifier.predict(me)
# print(predict_me)

percent = accuracy_score(y_test, y_pred) * 100
print(percent)
print(confusion_matrix(y_test, y_pred))

