import Preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

#Train model with Logistic Regression
X_train,X_test,y_train,y_test = Preprocessing.X_train,Preprocessing.X_test,Preprocessing.y_train,Preprocessing.y_test

#Using hyperparametter tunning
param_grid={'C':[0.001,0.01,0.1,1,10,50], 'max_iter':[100,200,300,400,500,700]}
log=RandomizedSearchCV(LogisticRegression(solver='lbfgs'),param_grid,cv=5)
log.fit(X_train,y_train)
y_hat_log=log.predict(X_test)
# print('Accuracy on train set: ',accuracy_score(y_train,log.predict(X_train)))
# print('Accuracy on test set: ',accuracy_score(y_test,y_hat_log))
# Accuracy on train set:  0.7204523984465204
# Accuracy on test set:  0.7198922426988306