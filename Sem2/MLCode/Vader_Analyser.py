# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:59:42 2020

@author: deepy
"""


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

import Pre_Processing_Text
def sentiment_scores(sentence): 

	# Create a SentimentIntensityAnalyzer object. 
	sid_obj = SentimentIntensityAnalyzer() 
	vp='undefined'
	fig=0
	
	preprocessed_text=Pre_Processing_Text.pre_process(sentence)
	
	print('preprocessed_text='+str(preprocessed_text))
	sentiment_dict = sid_obj.polarity_scores(preprocessed_text) 

	fig=sentiment_dict['compound']
	# decide sentiment as positive, negative and neutral 
	if sentiment_dict['compound'] >= 0.05 : 
#		print("Positive") 
		vp='positive'
	elif sentiment_dict['compound'] <= - 0.05 : 
#		print("Negative") 
		vp='negative'
	else : 
#		print("Neutral") 
		vp='neutral'
	print("vp",vp," fig",fig) 
	return vp,fig

