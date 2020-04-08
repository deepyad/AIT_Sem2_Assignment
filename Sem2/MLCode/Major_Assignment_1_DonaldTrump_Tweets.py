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
import TextBlob_Analyser
class TwitterClient(object): 

	def __init__(self): 

		consumer_key = '3H14qCCxj6ghzgSfbYjY7A5KL'
		consumer_secret = 'AkLnpLFbStlsUJBYBLcgDDr00ffxLWzd7ymFeHfdLmY5NOfD7B'
		access_token = '2358158900-nsAfJDYq1wkuHXxhQKM0pK3LqqX3ZIdBQK1q11e'
		access_token_secret = 'u8Lte5Yl14uldQpWkhHGrWY8uSiyDHzBgV56p068b3Jun'

		try: 
			self.auth = OAuthHandler(consumer_key, consumer_secret) 
			self.auth.set_access_token(access_token, access_token_secret) 
			self.api = tweepy.API(self.auth) 
			checking=self.api.user_timeline(screen_name = 'Donald J. Trump',count=200)
			print("Checking=>",len(checking)) 
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
		analysis = TextBlob(self.clean_tweet(tweet))  
		if analysis.sentiment.polarity > 0: 
			return 'positive'
		elif analysis.sentiment.polarity == 0: 
			return 'neutral'
		else: 
			return 'negative'


	def get_tweets(self, query, count = 500): 

		tweets = [] 
		try: 
			# call twitter api to fetch tweets 
			fetched_tweets = self.api.search(q = query, count = count) 
			for tweet in fetched_tweets: 
				parsed_tweet = {} 
				parsed_tweet['text'] = str(tweet.text )
				parsed_tweet['sentiment'] = str(self.get_tweet_sentiment(str(tweet.text)) )

				# appending parsed tweet to tweets list 
				if tweet.retweet_count > 0: 
					# if tweet has retweets, ensure that it is appended only once 
					if parsed_tweet not in tweets: 
						tweets.append(parsed_tweet) 
				else: 
					tweets.append(parsed_tweet) 
			return tweets 

		except tweepy.TweepError as e: 
			print("Error : " + str(e)) 


	def read_tweets(self): 

		tweets = [] 
		
		try: 
			# call twitter api to fetch tweets 
            #trump_data_unique_joined.csv
			with open('trump_data_unique_joined.csv', 'r',encoding="utf-8") as file:
				reader = csv.reader(file)
				for row in reader:
					#print(row)
				    parsed_tweet = {} 
				#read_tweets = self.api.search(q = query, count = count) 
			#for tweet in fetched_tweets: 
			#	
				    parsed_tweet['text'] = str(row )
				    parsed_tweet['sentiment'] = str(self.get_tweet_sentiment(str(row)) )
				#	print('parsed_tweet[text] ',parsed_tweet['text'])
				#	print('parsed_tweet[sentiment]',parsed_tweet['sentiment'])
				# appending parsed tweet to tweets list 
			#	if tweet.retweet_count > 0: 
					# if tweet has retweets, ensure that it is appended only once 
				    if parsed_tweet not in tweets: 
				    				    tweets.append(parsed_tweet) 
                    
				#	else: 
				#		tweets.append(parsed_tweet) 
				#for x in tweets:
				#	print('<=>',x)
			return tweets 

		except tweepy.TweepError as e: 
			print("Error : " + str(e)) 


def main(): 
 
	api = TwitterClient() 
	#web_tweets=api.user_timeline(screen_name = 'Donald J. Trump',count=200)
	web_tweets = api.get_tweets(query = 'Donald J. Trump', count = 5000000) 
	file_tweets=api.read_tweets()
	tweets=web_tweets+file_tweets
	#tweets.append(web_tweets)
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
	neutweets = [tweet for tweet in tweets if tweet['sentiment'] == 'neutral'] 

	print("\n\n ********** TextBlob labelled Positive tweets:") 
	for tweet in ptweets[:10]: 
		print('Tweet=> ',tweet['text'])
		print('---Vader Analysis of above positive(as per TextBlob\'s classification) tweet---')
		print(Vader_Analyser.sentiment_scores(tweet['text'])) 
		print('----')	
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
		print('----')	
	
	with open(r'TweetsFinalAnalysis.csv', 'w', newline='',encoding="utf-8") as csvfile:
		fieldnames = ['Tweet','TextBlob Polarity','TextBlob Score','Vader Polarity','Vader Score']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()	 
		for tweet in ptweets: 
			va=Vader_Analyser.sentiment_scores(tweet['text'])
			writer.writerow({'Tweet':tweet['text'], 'TextBlob Polarity': 'Positive','TextBlob Score':str(round(TextBlob_Analyser.get_tweet_sentiment_score(tweet['text']),2)),'Vader Polarity':va[0],'Vader Score':round(va[1],2)})
		for tweet in ntweets: 
			va=Vader_Analyser.sentiment_scores(tweet['text'])
			writer.writerow({'Tweet':tweet['text'], 'TextBlob Polarity':'Negative','TextBlob Score':str(round(TextBlob_Analyser.get_tweet_sentiment_score(tweet['text']),2)),'Vader Polarity':va[0],'Vader Score':round(va[1],2)})
		for tweet in neutweets: 
			va=Vader_Analyser.sentiment_scores(tweet['text'])
			writer.writerow({'Tweet':tweet['text'], 'TextBlob Polarity':'Neutral','TextBlob Score':str(round(TextBlob_Analyser.get_tweet_sentiment_score(tweet['text']),2)),'Vader Polarity':va[0],'Vader Score':round(va[1],2)})

	print('Size of tweets:=',len(tweets))
	print('TextBlob Analysis overall %')
	print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
	print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))) 
	print("Neutral tweets percentage: {} %".format(100*len(neutweets)/len(tweets))) 

if __name__ == "__main__": 
	main() 
