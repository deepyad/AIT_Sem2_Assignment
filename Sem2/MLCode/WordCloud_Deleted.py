# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:34:18 2020

@author: deepy
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:35:48 2020

@author: deepy
"""

import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt


df = pd.read_csv('TweetsFinalAnalysis_BI.csv', sep=',')

df.head()

#text = df.FilteredTweet[0]
text = " ".join(review for review in df.FilteredTweet)
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#wordcloud.to_file("img/first_review.png")


print ("There are {} words in the combination of all review.".format(len(text)))


# Create stopword list:
#stopwords = set(STOPWORDS)
#stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

# Generate a word cloud image
#wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#wine_mask = np.array(Image.open("DonaldTrump.png"))

# Transform your mask into a new one that will work with the function:
#transformed_wine_mask = np.ndarray((wine_mask.shape[0],wine_mask.shape[1]), np.int32)

def transform_format(val):
     if val !=0:
         return val
     else:
         return 255
''' if val == 0:
        return 255
    else:
        return val
'''

#for i in range(len(wine_mask)):
#    transformed_wine_mask[i] = list(map(transform_format, wine_mask[i]))
    
#transformed_wine_mask
#------------------------------------------
# Generate a word cloud image
#mask = np.array(Image.open("DonaldTrump.png"))
#wordcloud_usa = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA", max_words=1000, mask=mask).generate(text)

# create coloring from image
#image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[7,7])
#plt.imshow(wordcloud_usa.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

# store to file
plt.savefig("Donald_Trump_created.png", format="png")

plt.show()