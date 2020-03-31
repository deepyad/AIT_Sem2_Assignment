# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:12:06 2020

@author: deepy
"""

#import pickle, gzip
#with gzip.open('Data/20newsgroups.pkl.gz', 'rb') as f:
    #newsgroups_train = pickle.load(f)
    
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text

data=fetch_20newsgroups()
print('Categories= ',data['target_names'])
categories=data['target_names']
twenty_train = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories,shuffle=True, random_state=42)

my_stop_words = text.ENGLISH_STOP_WORDS

#Unique number is given to every word and then it is counted how many times that number is appearing.
#Countvector class is used for this

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

text_clf=Pipeline([('vect',TfidfVectorizer(ngram_range=(1,1), stop_words=my_stop_words)),('clf',MultinomialNB())])
text_clf.fit(twenty_train.data,twenty_train.target)

predicted= text_clf.predict(twenty_test.data)

from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np
print("Accuracy achieved is :",np.mean(predicted==twenty_test.target))
print(metrics.classification_report(twenty_test.target,predicted,target_names=twenty_test.target_names)),
metrics.confusion_matrix(twenty_test.target,predicted)

