# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:29:53 2020

@author: Deepak Yadav
"""
import csv
import tweepy 
from tweepy import OAuthHandler 
import Vader_Analyser
import TextBlob_Analyser
import Pre_Processing_Text
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
		#	print("Checking=>",len(checking)) 
		except: 
			print("Error: Authentication Failed") 

	def get_tweets_from_web(self, query, count = 500): 

		tweets = [] 
		try: 
			# call twitter api to fetch tweets 
			fetched_tweets = self.api.search(q = query, count = count) 
			for tweet in fetched_tweets: 
				parsed_tweet = {} 
				parsed_tweet['text'] = str(tweet.text )
				parsed_tweet['sentiment'] = str(TextBlob_Analyser.get_tweet_sentiment(str(tweet.text)) )

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


	def read_tweets_from_file(self): 

		tweets = [] 
		
		try: 
			# call twitter api to fetch tweets 
            #trump_data_unique_joined.csv
			with open('trump_data_unique_joined_Removed_PosNegNuet.csv', 'r',encoding="utf-8") as file:
				reader = csv.reader(file)
				next(reader, None)
				for row in reader:
				    parsed_tweet = {}
				    parsed_tweet['text'] = str(row )
				    parsed_tweet['sentiment'] = str(TextBlob_Analyser.get_tweet_sentiment(str(row)) )
				    #print('parsed_tweet[0]',parsed_tweet['text'])
				   #print('parsed_tweet[1]',parsed_tweet['sentiment'])
				    if parsed_tweet not in tweets: 
				    				    tweets.append(parsed_tweet) 
			return tweets 

		except tweepy.TweepError as e: 
			print("Error : " + str(e)) 


def main(): 
 
	api = TwitterClient() 
	web_tweets = api.get_tweets_from_web(query = 'Trump', count = 5000000) 
	file_tweets=api.read_tweets_from_file()
	tweets=web_tweets+file_tweets

	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
	neutweets = [tweet for tweet in tweets if tweet['sentiment'] == 'neutral'] 
	
	with open(r'TweetsFinalAnalysis.csv', 'w', newline='',encoding="utf-8") as csvfile:
		fieldnames = ['Tweet','FilteredTweet','TextBlob Polarity','TextBlob Score','Vader Polarity','Vader Score']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()	 
		for tweet in ptweets: 
			va=Vader_Analyser.sentiment_scores(tweet['text'])
			writer.writerow({'Tweet':tweet['text'],'FilteredTweet':Pre_Processing_Text.pre_process(tweet['text']), 'TextBlob Polarity': 'Positive','TextBlob Score':str(round(TextBlob_Analyser.get_tweet_sentiment_score(tweet['text']),2)),'Vader Polarity':va[0],'Vader Score':round(va[1],2)})
		for tweet in ntweets: 
			va=Vader_Analyser.sentiment_scores(tweet['text'])
			writer.writerow({'Tweet':tweet['text'],'FilteredTweet':Pre_Processing_Text.pre_process(tweet['text']),'TextBlob Polarity':'Negative','TextBlob Score':str(round(TextBlob_Analyser.get_tweet_sentiment_score(tweet['text']),2)),'Vader Polarity':va[0],'Vader Score':round(va[1],2)})
		for tweet in neutweets: 
			va=Vader_Analyser.sentiment_scores(tweet['text'])
			writer.writerow({'Tweet':tweet['text'],'FilteredTweet':Pre_Processing_Text.pre_process(tweet['text']), 'TextBlob Polarity':'Neutral','TextBlob Score':str(round(TextBlob_Analyser.get_tweet_sentiment_score(tweet['text']),2)),'Vader Polarity':va[0],'Vader Score':round(va[1],2)})

	print('Size of tweets:=',len(tweets))
	print('Tweets Read from the Web:=',len(web_tweets))
	print('Tweets Read from the File:=',len(file_tweets))
	print('Size of TextBlob Positive tweets:=',len(ptweets))
	print('Size of TextBlob Negative tweets:=',len(ntweets))
	print('Size of TextBlob Neutral tweets:=',len(neutweets))
	print("TextBlob Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
	print("TextBlob Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))) 
	print("TextBlob Neutral tweets percentage: {} %".format(100*len(neutweets)/len(tweets))) 

if __name__ == "__main__": 
	main() 
