import Preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Train model with KNN
X_train,X_test,y_train,y_test = Preprocessing.X_train,Preprocessing.X_test,Preprocessing.y_train,Preprocessing.y_test

#Find k in range value
def Knn(x,y,left,right):
    knn_score = []
    for i in range (left,right):
        knn = KNeighborsClassifier(n_neighbors = i,weights = 'uniform')
        knn.fit(x,y)
        y_hat = knn.predict(X_test)
        knn_score.append(accuracy_score(y_hat,y_test))
    return knn_score

# After run knn function, choose k = 65 for training
knn=KNeighborsClassifier(n_neighbors=65)
knn.fit(X_train,y_train)
y_hat = knn.predict(X_test)
# print('Accuracy on train set: {}'.format(accuracy_score(y_train,knn.predict(X_train))))
# print('Accuracy on test set: {}'.format(accuracy_score(y_hat,y_test)))

# Accuracy on train set: 0.7257531227038942
# Accuracy on test set: 0.7170758586909937

