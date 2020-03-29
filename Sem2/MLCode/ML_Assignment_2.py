# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:00:26 2020

@author: deepy
"""
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.datasets import fetch_20newsgroups
data=fetch_20newsgroups()

datasets.fetch_20newsgroups()
#data1=pd.DataFrame(data= np.c_[fetcheddata['data'], fetcheddata['target']],
#                     columns= fetcheddata['feature_names'] + ['target'])

dataset = fetch_20newsgroups()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data.to_csv('20_newsgroup.csv')
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
df.columns = ['text', 'target']

targets = pd.DataFrame( newsgroups_train.target_names)
targets.columns=['title']

out = pd.merge(df, targets, left_on='target', right_index=True)
out['date'] = pd.to_datetime('now')
out.to_csv('20_newsgroup.csv')
#print(boston_data.feature_names)
#df_boston = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
#df_boston['target'] = pd.Series(boston_data.target)
#print(df_boston.head())



#df = pd.DataFrame(data.data, columns=data.key)


#print(data.values())
#data.Sentiment.value_counts()