import pandas as pd

all_tweets = pd.read_json("random_tweets.json", lines=True)

print(len(all_tweets))
print(all_tweets.columns)
print(all_tweets.loc[0]['text'])

#Print the user here and the user's location here.
print (all_tweets.loc[0]["user"]["location"])

import numpy as np
all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] > 5,1,0)

print (all_tweets["retweet_count"].median())
print (all_tweets["is_viral"].value_counts())

all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)

from sklearn.preprocessing import scale
labels = all_tweets['is_viral']
data = all_tweets[["tweet_length", "followers_count", "friends_count"]]

scaled_data = scale(data, axis = 0)
print (scaled_data[0])

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state = 1)

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
score = []
for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(train_data, train_labels)
    score.append(classifier.score(test_data, test_labels))

plt.plot(range(1,200), score)
plt.show()