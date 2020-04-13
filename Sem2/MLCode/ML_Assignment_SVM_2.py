# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:31:41 2020

@author: deepy
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer

data=fetch_20newsgroups()
print('Categories= ',data['target_names'])
categories=data['target_names']
twenty_train = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories,shuffle=True, random_state=42)

my_stop_words = text.ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

text_clf=Pipeline([('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', SGDClassifier(loss='hinge', penalty='l2',
 alpha=1e-3, random_state=42,
 max_iter=5, tol=None))])

text_clf.fit(twenty_train.data,twenty_train.target)
predicted= text_clf.predict(twenty_test.data)

from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np
print("Accuracy achieved is :",np.mean(predicted==twenty_test.target))
print(metrics.classification_report(twenty_test.target,predicted,target_names=twenty_test.target_names)),
metrics.confusion_matrix(twenty_test.target,predicted)

