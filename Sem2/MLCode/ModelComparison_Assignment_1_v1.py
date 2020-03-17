# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:47:25 2020

@author: deepy
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Load data
df = pd.read_csv('Data\Frogs_MFCCs_KNN_data.csv')

print(df.shape)
df.head()

df.drop(['Genus', 'Species','RecordID'], axis=1, inplace=True)

y = df['Family'].values
x = df.drop('Family', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=10)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
 
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print((156+562+1276)/(22+188+636+1313))
print("Regression Analysys Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, average='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, average ='weighted'))
#https://intellipaat.com/community/12741/sklearn-metrics-for-multiclass-classification
#The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
#The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
# Bagged Decision Trees for Classification
kfold = model_selection.KFold(n_splits=10, random_state=10)
model_1 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=10)
model_1.fit(X_train,y_train)
y_pred=model_1.predict(X_test)
results_1 = model_selection.cross_val_score(model_1, x, y, cv=kfold)
print('Bagged Decision Mean',results_1.mean())
print("Bagged Decision Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred,average ='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred,average ='weighted'))

# Random Forest Classification
kfold_rf = model_selection.KFold(n_splits=10, random_state=10)
model_rf = RandomForestClassifier(n_estimators=100, max_features=5)
model_rf.fit(X_train,y_train)
y_pred=model_rf.predict(X_test)
results_rf = model_selection.cross_val_score(model_rf, x, y, cv=kfold_rf)
print('Random Forest Mean',results_rf.mean())
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, average ='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, average ='weighted'))

# Boosting Classification
from sklearn.ensemble import AdaBoostClassifier
kfold_ada = model_selection.KFold(n_splits=10, random_state=10)
model_ada = AdaBoostClassifier(n_estimators=30, random_state=10)
model_ada.fit(X_train,y_train)
y_pred=model_rf.predict(X_test)
results_ada = model_selection.cross_val_score(model_ada, x, y, cv=kfold_ada)
print('Boosting Mean',results_ada.mean())
print("Boosting Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, average ='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred,average ='weighted'))

# SVM Classification
model_svm = svm.SVC(kernel='linear') 
model_svm.fit(X_train,y_train)
y_pred=model_svm.predict(X_test)
#results_ada = model_selection.cross_val_score(model_ada, x, y, cv=kfold_ada)
#print('Boosting Mean',results_ada.mean())
print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, average ='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred,average ='weighted'))

#Voting Classifier
kfold_vc = model_selection.KFold(n_splits=10, random_state=10)
estimators = []
mod_lr = LogisticRegression()
estimators.append(('logistic', mod_lr))
mod_dt = DecisionTreeClassifier()
estimators.append(('cart', mod_dt))
mod_sv = SVC()
estimators.append(('svm', mod_sv))
# Lines 9 to 11
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,y_train)
y_pred=ensemble.predict(X_test)
results_vc = model_selection.cross_val_score(ensemble, x, y, cv=kfold_vc)
print('Voting Classifier Mean ',results_vc.mean())
print("Voting Classifier Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Neuron Network Classifier
#https://www.pluralsight.com/guides/machine-learning-neural-networks-scikit-learn
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print('Below is Train Section')
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))
print('Below is Test Section')
print(confusion_matrix(y_test,predict_test))
print('Final Figures of Accuracy')
print(classification_report(y_test,predict_test))
print(metrics.accuracy_score(y_test,predict_test))