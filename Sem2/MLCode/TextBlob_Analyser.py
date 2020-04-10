# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:35:12 2020

@author: deepy
"""
from textblob import TextBlob 
import Pre_Processing_Text


def get_tweet_sentiment_score(tweet): 
		''' 
		Utility function to classify sentiment of passed tweet 
		using textblob's sentiment method 
		'''
		analysis = TextBlob(Pre_Processing_Text.pre_process(tweet)) 
		return analysis.sentiment.polarity

def get_tweet_sentiment(tweet): 
		''' 
		Utility function to classify sentiment of passed tweet 
		using textblob's sentiment method 
		'''
		analysis = TextBlob(Pre_Processing_Text.pre_process(tweet))  
		if analysis.sentiment.polarity > 0: 
			return 'positive'
		elif analysis.sentiment.polarity == 0: 
			return 'neutral'
		else: 
			return 'negative'
