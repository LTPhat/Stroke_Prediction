import Preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#Train model with SVM
X_train,X_test,y_train,y_test = Preprocessing.X_train,Preprocessing.X_test,Preprocessing.y_train,Preprocessing.y_test
support_vector = SVC(gamma ='auto', probability = True)
support_vector.fit(X_train,y_train)
y_pred_sup = support_vector.predict(X_test)
# print('Accuracy on train set: ',accuracy_score(y_train,support_vector.predict(X_train)))
# print('Accuracy on test set: ',accuracy_score(y_test,y_pred_sup))
# Accuracyon train set:  0.725149574892411
# Accuracy on test set:  0.7222188207922611