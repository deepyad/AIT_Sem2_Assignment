# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:35:12 2020

@author: deepy
"""
from textblob import TextBlob 
import re 

def clean_tweet(tweet): 
		''' 
		Utility function to clean tweet text by removing links, special characters 
		using simple regex statements. 
		'''
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split()) 



def get_tweet_sentiment_score(tweet): 
		''' 
		Utility function to classify sentiment of passed tweet 
		using textblob's sentiment method 
		'''
		analysis = TextBlob(clean_tweet(tweet)) 
		return analysis.sentiment.polarity