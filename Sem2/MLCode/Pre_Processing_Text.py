# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:52:23 2020

@author: Deepak Yadav
"""
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def remove_stop_words(sentence):
    
    return sentence

def clean_text(text): 
		''' 
		Utility function to clean tweet text by removing links, special characters 
		using simple regex statements. 
		'''
		emails_removed=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+) | (r'\b\d+(?:\.\d+)?\s+')", " ", text).split()) 
		numbers_removed=''.join(i for i in emails_removed if not i.isdigit())
		special_chars_removed=re.sub('[^A-Za-z0-9]+', ' ', numbers_removed)
		#print('Cleaned text=>',special_chars_removed)
		return special_chars_removed 

def pre_process(sentence):
 #   print('Sourced text =',sentence)
    cleaned_text=clean_text(sentence)
 #   print('cleaned_text=',cleaned_text)
    tokenized_word=word_tokenize(cleaned_text)
   # print('tokenized_word=',tokenized_word)
    stop_words=set(stopwords.words("english"))
    filtered=[]
    for w in tokenized_word:
       if w not in stop_words:
           filtered.append(w)


    filtered=' '.join(filtered)
   # print('filtered_sent=',filtered_sent)
    return filtered