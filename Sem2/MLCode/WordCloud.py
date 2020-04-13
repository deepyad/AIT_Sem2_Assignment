# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:35:48 2020

@author: Deepak Yadav
"""

import pandas as pd
from wordcloud import WordCloud

import matplotlib.pyplot as plt


df = pd.read_csv('TweetsFinalAnalysis.csv', sep=',')

df.head()

text = " ".join(review for review in df.FilteredTweet)
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig("WordCloud.jpg", format="jpg")
plt.show()

print ("There are {} words in the combination of all review.".format(len(text)))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.figure(figsize=[7,7])
plt.axis("off")

plt.savefig("Donald_Trump_created.png", format="png")
plt.show()