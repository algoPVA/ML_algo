from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from prepare_data1 import x_train,y_train,x_test,y_test

classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

percent = accuracy_score(y_test, y_pred) * 100
print(percent)
print(confusion_matrix(y_test, y_pred))