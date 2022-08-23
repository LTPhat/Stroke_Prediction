# Import packages

import Preprocessing
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score

#Import file

import KNN
import LogisticRegression
import SVM
import GradientBoost

X_train,X_test,y_train,y_test = Preprocessing.X_train,Preprocessing.X_test,Preprocessing.y_train,Preprocessing.y_test

# Choose 4 best model to combine
knn = KNN.knn
log = LogisticRegression.log
svm = SVM.support_vector
gbc = GradientBoost.gbc



#Voting Classifier

from sklearn.ensemble import VotingClassifier
clf1 = knn
clf2 = log
clf3 = gbc
clf4 = svm

eclf = VotingClassifier(
    estimators=[("knn", clf1), ("log", clf2), ("gbc", clf3),('svm',clf4)],
    voting="soft",
    weights=[2,1,2,2],
)
eclf.fit(X_train, y_train)

import pickle
pickle.dump(eclf,open('model.pkl','wb'))
