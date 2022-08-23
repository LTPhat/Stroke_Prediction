import Preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
#Train model with KNN
X_train,X_test,y_train,y_test = Preprocessing.X_train,Preprocessing.X_test,Preprocessing.y_train,Preprocessing.y_test


param_grid = {
    'learning_rate' : [0.05, 0.075, 0.1, 0.25],
    'n_estimators': [50, 75,100, 150],
}
gbc=RandomizedSearchCV(GradientBoostingClassifier(random_state=42),param_grid,cv=5)
gbc.fit(X_train,y_train)
y_pred_gbc=gbc.predict(X_test)
# print('Accuracy on train set: {}'.format(accuracy_score(y_train,gbc.predict(X_train))))
# print('Accuracy on test set: {}'.format(accuracy_score(y_pred_gbc,y_test)))
# Accuracy on train set: 0.7277999370210979
# Accuracy on test set: 0.728096491765138
