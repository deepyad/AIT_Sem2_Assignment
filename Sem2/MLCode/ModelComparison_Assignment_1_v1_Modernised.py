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
from sklearn.model_selection import GridSearchCV

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
df.drop(['Genus', 'Species','RecordID'], axis=1, inplace=True)
print('Rows and columns in complete dataframe : ',df.shape)

y = df['Family'].values
x = df.drop('Family', axis=1).values

#Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=10)
#Creating Linear regression model
logreg = LogisticRegression()
#Traiaing the model
logreg.fit(X_train, y_train)
#Predicting output from trained model
y_pred = logreg.predict(X_test)
print('************Linear Regression Analysis Section****************')
print('Confusion Matrix in case of Linear Regression Model') 
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
print(classification_report(y_test, y_pred))
#metrics.f1_score(y_test, y_pred, labels=np.unique(y_pred),average ='weighted')
#(_, _, f1, _) = metrics.precision_recall_fscore_support(y_test, y_pred,average='weighted',warn_for=tuple())
print("Linear Regression Analysys Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Linear Regression Analysis Precision:",metrics.precision_score(y_test, y_pred, average='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Linear Regression Analysis Recall:",metrics.recall_score(y_test, y_pred, average ='weighted'))

print('************Bagged Decision Analysis Section****************')
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
print("Bagged Decision Precision:",metrics.precision_score(y_test, y_pred,average ='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Bagged Decision Recall:",metrics.recall_score(y_test, y_pred,average ='weighted'))

print('************Random Forest Analysis Section****************')
# Random Forest Classification
kfold_rf = model_selection.KFold(n_splits=10, random_state=10)
model_rf = RandomForestClassifier(n_estimators=100, max_features=5)
model_rf.fit(X_train,y_train)
y_pred=model_rf.predict(X_test)
results_rf = model_selection.cross_val_score(model_rf, x, y, cv=kfold_rf)
print('Random Forest Mean',results_rf.mean())
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Random Forest Precision:",metrics.precision_score(y_test, y_pred, average ='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Random Forest Recall:",metrics.recall_score(y_test, y_pred, average ='weighted'))

print('************Ada Boosting Analysis Section****************')
# Boosting Classification
from sklearn.ensemble import AdaBoostClassifier
kfold_ada = model_selection.KFold(n_splits=10, random_state=10)
model_ada = AdaBoostClassifier(n_estimators=30, random_state=10)
model_ada.fit(X_train,y_train)
y_pred=model_rf.predict(X_test)
results_ada = model_selection.cross_val_score(model_ada, x, y, cv=kfold_ada)
print('Ada Boosting Mean',results_ada.mean())
print("Ada Boosting Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Ada Boosting Precision:",metrics.precision_score(y_test, y_pred, average ='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Ada Boosting Recall:",metrics.recall_score(y_test, y_pred,average ='weighted'))

print('************SVM Analysis Section****************')
# SVM Classification
model_svm = svm.SVC(kernel='linear') 
model_svm.fit(X_train,y_train)
y_pred=model_svm.predict(X_test)
#results_ada = model_selection.cross_val_score(model_ada, x, y, cv=kfold_ada)
#print('Boosting Mean',results_ada.mean())
print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("SVM Precision:",metrics.precision_score(y_test, y_pred, average ='weighted'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("SVM Recall:",metrics.recall_score(y_test, y_pred,average ='weighted'))

print('************Voting Classifier Analysis Section****************')
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
print("Voting Classifier Recall:",metrics.recall_score(y_test, y_pred,average ='weighted'))

print('************Neuoron Network Analysis(Default Parameters) Section****************')
#Neuron Network Classifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print('Neuron Network Train Section Confusion Matrix')
print(confusion_matrix(y_train,predict_train))
print('Neuron Network Trainaing Data Classification Report')
print(classification_report(y_train,predict_train))
print('Neuron Network Test Section Confusion Matrix')
print(confusion_matrix(y_test,predict_test))
print('Neuron Network Test Data Classification Report')
print(classification_report(y_test,predict_test))
print('Neuron Network Accuracy :',metrics.accuracy_score(y_test,predict_test))

print('current loss computed with the loss function: ',mlp.loss_)
#print('coefs: ', mlp.coefs_)
#print('intercepts: ',mlp.intercepts_)
print(' number of iterations the solver: ', mlp.n_iter_)
print('num of layers: ', mlp.n_layers_)
print('Num of o/p: ', mlp.n_outputs_)
print('************ Identifying appropriate prarmeters ****************')
#param_grid = [
#        {
#            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
#            'solver' : ['lbfgs', 'sgd', 'adam'],
#            'hidden_layer_sizes': [
#             (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
#             ]
#        }
#       ]
#mlp = GridSearchCV(MLPClassifier(), param_grid, cv=3, scoring='accuracy')
#mlp.fit(X_train,y_train)


#print("Best parameters set found on development set:")
#print(mlp.best_params_)
#Best parameters set found on development set:
#{'activation': 'logistic', 'hidden_layer_sizes': (21,), 'solver': 'lbfgs'}
#
#Neuron Network Classifier
print('************ Neuron Network Analysis using above optimised parameters ****************')
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(21,21,21), activation='logistic', solver='lbfgs', max_iter=500)
mlp.fit(X_train,y_train)
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print('Neuron Network Train Section Confusion Matrix')
print(confusion_matrix(y_train,predict_train))
print('Neuron Network Trainaing Data Classification Report')
print(classification_report(y_train,predict_train))
print('Neuron Network Test Section Confusion Matrix')
print(confusion_matrix(y_test,predict_test))
print('Neuron Network Test Data Classification Report')
print(classification_report(y_test,predict_test))
print('Neuron Network Accuracy :',metrics.accuracy_score(y_test,predict_test))
print('current loss computed with the loss function: ',mlp.loss_)
#print('coefs: ', mlp.coefs_)
#print('intercepts: ',mlp.intercepts_)
print(' number of iterations the solver: ', mlp.n_iter_)
print('num of layers: ', mlp.n_layers_)
print('Num of o/p: ', mlp.n_outputs_)
print('************Analysis Ends****************')

