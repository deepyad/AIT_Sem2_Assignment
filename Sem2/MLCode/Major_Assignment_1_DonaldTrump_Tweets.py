# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:29:53 2020

@author: deepy
"""
import csv
import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
import Vader_Analyser
class TwitterClient(object): 
	''' 
	Generic Twitter Class for sentiment analysis. 
	'''
	def __init__(self): 
		''' 
		Class constructor or initialization method. 
		'''
		# keys and tokens from the Twitter Dev Console 
		consumer_key = '3H14qCCxj6ghzgSfbYjY7A5KL'
		consumer_secret = 'AkLnpLFbStlsUJBYBLcgDDr00ffxLWzd7ymFeHfdLmY5NOfD7B'
		access_token = '2358158900-nsAfJDYq1wkuHXxhQKM0pK3LqqX3ZIdBQK1q11e'
		access_token_secret = 'u8Lte5Yl14uldQpWkhHGrWY8uSiyDHzBgV56p068b3Jun'

		# attempt authentication 
		try: 
			# create OAuthHandler object 
			self.auth = OAuthHandler(consumer_key, consumer_secret) 
			# set access token and secret 
			self.auth.set_access_token(access_token, access_token_secret) 
			# create tweepy API object to fetch tweets 
			self.api = tweepy.API(self.auth) 
		except: 
			print("Error: Authentication Failed") 

	def clean_tweet(self, tweet): 
		''' 
		Utility function to clean tweet text by removing links, special characters 
		using simple regex statements. 
		'''
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split()) 

	def get_tweet_sentiment(self, tweet): 
		''' 
		Utility function to classify sentiment of passed tweet 
		using textblob's sentiment method 
		'''
		# create TextBlob object of passed tweet text 
		analysis = TextBlob(self.clean_tweet(tweet)) 
		# set sentiment 
		if analysis.sentiment.polarity > 0: 
			return 'positive'
		elif analysis.sentiment.polarity == 0: 
			return 'neutral'
		else: 
			return 'negative'

	def get_tweets(self, query, count = 10): 
		''' 
		Main function to fetch tweets and parse them. 
		'''
		# empty list to store parsed tweets 
		tweets = [] 

		try: 
			# call twitter api to fetch tweets 
			fetched_tweets = self.api.search(q = query, count = count) 

			# parsing tweets one by one 
			for tweet in fetched_tweets: 
				# empty dictionary to store required params of a tweet 
				parsed_tweet = {} 

				# saving text of tweet 
				parsed_tweet['text'] = tweet.text 
				# saving sentiment of tweet 
				parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 

				# appending parsed tweet to tweets list 
				if tweet.retweet_count > 0: 
					# if tweet has retweets, ensure that it is appended only once 
					if parsed_tweet not in tweets: 
						tweets.append(parsed_tweet) 
				else: 
					tweets.append(parsed_tweet) 

			# return parsed tweets 
			return tweets 

		except tweepy.TweepError as e: 
			# print error (if any) 
			print("Error : " + str(e)) 

def main(): 
	# creating object of TwitterClient Class 
	api = TwitterClient() 
	# calling function to get tweets 
	tweets = api.get_tweets(query = 'Donald J. Trump', count = 50000) 

	# picking positive tweets from tweets 
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
	# percentage of positive tweets 
	print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
	# picking negative tweets from tweets 
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
	# percentage of negative tweets 
	print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))) 
    # picking negative tweets from tweets 
	neutweets = [tweet for tweet in tweets if tweet['sentiment'] == 'neutral'] 
	print("Neutral tweets percentage: {} %".format(100*len(neutweets)/len(tweets))) 
	
    

	 #writer.writerow({'TextBlob Polarity':'Positive'})
    
	# printing first 5 positive tweets 
	print("\n\n ********** TextBlob labelled Positive tweets:") 
	for tweet in ptweets[:10]: 
		print('Tweet=> ',tweet['text'])
		print('---Vader Analysis of above positive(as per TextBlob\'s classification) tweet---')
		print(Vader_Analyser.sentiment_scores(tweet['text'])) 
		#writer.writerows({'Tweet':tweet['text'], 'TextBlob Polarity':'Positive','Vader Polarity':'Negative'})
		#writer.writerow({'TextBlob Polarity':'Positive'})
        
		print('----')	
    # printing first 5 negative tweets 
	print("\n\n ********** TextBlob labelled Negative tweets:") 
	for tweet in ntweets[:10]: 
		print('Tweet=> ',tweet['text']) 
		print('---Vader Analysis of above negative(as per TextBlob\'s classification) tweet---')
		print(Vader_Analyser.sentiment_scores(tweet['text']))
		print('----')	        
	print("\n\n ********** TextBlob labelled Neutral tweets:") 
	for tweet in neutweets[:10]: 
		print('Tweet=> ',tweet['text'])
		print('---Vader Analysis of above neutral(as per TextBlob\'s classification) tweet---')
		print(Vader_Analyser.sentiment_scores(tweet['text']))
#		print(tweet['text'])
	#print(Vader_Analyser.sentiment_scores(tweet['text']))
		print('----')	
	
	with open(r'Tweetfile.csv', 'w', newline='',encoding="utf-8") as csvfile:
		fieldnames = ['Tweet','TextBlob Polarity','Vader Polarity']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()	 
		for tweet in ptweets[:100]: 
			writer.writerow({'Tweet':tweet['text'], 'TextBlob Polarity':'Positive','Vader Polarity':Vader_Analyser.sentiment_scores(tweet['text'])})
		for tweet in ntweets[:100]: 
			writer.writerow({'Tweet':tweet['text'], 'TextBlob Polarity':'Negative','Vader Polarity':Vader_Analyser.sentiment_scores(tweet['text'])})
		for tweet in neutweets[:100]: 
			writer.writerow({'Tweet':tweet['text'], 'TextBlob Polarity':'Neutral','Vader Polarity':Vader_Analyser.sentiment_scores(tweet['text'])})

if __name__ == "__main__": 
	# calling main function 
	main() 
