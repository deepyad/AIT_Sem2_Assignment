# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 21:14:29 2020

@author: deepy
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text import FreqDistVisualizer
from yellowbrick.datasets import load_hobbies
import numpy as np

data=pd.read_csv('TweetsFinalAnalysis.csv', sep=',')
data.head()

data.info()


Sentiment_count=data.groupby('TextBlob Score').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Filtered Tweet'])
plt.xlabel('TextBlob Score')
plt.ylabel('Number of Tweets')
plt.show()

Sentiment_count=data.groupby('TextBlob Polarity').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Filtered Tweet'])
plt.xlabel('TextBlob Polarity')
plt.ylabel('Number of Tweets')
plt.show()

Sentiment_count=data.groupby('Vader Score').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Filtered Tweet'])
plt.xlabel('Vader Score')
plt.ylabel('Number of Tweets')
plt.show()

Sentiment_count=data.groupby('Vader Polarity').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Filtered Tweet'])
plt.xlabel('TextBlob Polarity')
plt.ylabel('Number of Tweets')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Filtered Tweet'])

frequencies = sum(text_counts).toarray()[0]
df=pd.DataFrame(frequencies, index=cv.get_feature_names(), columns=['frequency'])
df.to_csv('Word_Frequency.csv')
print(df.describe())
#df.hist()
#ax = df.plot.bar(x='Index', y='frequency', rot=0)
df.plot()
plt.savefig("WordFrequencyChart.png", format="png")
#print('text_counts=',text_counts)
#print('cv.vocabulary_=',cv.vocabulary_)
vocab = cv.vocabulary_
tot = sum(vocab.values())
frequency = {vocab[w]/tot for w in vocab.keys()}
#print(frequency)
vocab = cv.vocabulary_
print('----')
print(cv.get_feature_names())
print(text_counts.toarray())


train_data_features = text_counts.toarray()
vocab = cv.get_feature_names()
dist = np.sum(train_data_features, axis=0)
ngram_freq = {}
print('ngram_freq=',ngram_freq)

# For each, print the vocabulary word and the frequency
for tag, count in zip(vocab, dist):
    #print(tag, count)
    ngram_freq[tag]=count
#corpus = load_hobbies()

#vectorizer = CountVectorizer()
#docs       = vectorizer.fit_transform(data)
#features   = vectorizer.get_feature_names()

#visualizer = FreqDistVisualizer(features=features, orient='v')
#visualizer.fit(docs)
#visualizer.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_counts, data['Filtered Tweet'], test_size=0.3, random_state=1)

from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(data['Filtered Tweet'])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_tf, data['Filtered Tweet'], test_size=0.3, random_state=123)


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))