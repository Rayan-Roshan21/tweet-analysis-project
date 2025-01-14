# Importing pandas library for data manipulation and analysis
import pandas as pd

# Reading JSON file containing tweets from New York, structured as one JSON object per line
new_york_tweets = pd.read_json("new_york.json", lines=True)
# Printing the number of tweets in the New York dataset
print(len(new_york_tweets))
# Printing the column names in the New York dataset
print(new_york_tweets.columns)
# Printing the text content of the tweet at index 12 in the New York dataset
print(new_york_tweets.loc[12]["text"])

# Reading JSON file containing tweets from London
london_tweets = pd.read_json("london.json", lines=True)
# Printing the number of tweets in the London dataset
print(len(london_tweets))

# Reading JSON file containing tweets from Paris
paris_tweets = pd.read_json("paris.json", lines=True)
# Printing the number of tweets in the Paris dataset
print(len(paris_tweets))

# Extracting the 'text' column (tweet text) as a list from each dataset
new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()

# Combining tweet text from all three cities into a single list
all_tweets = new_york_text + london_text + paris_text

# Creating labels for the tweets:
# 0 for New York, 1 for London, 2 for Paris
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)

# Importing train_test_split from scikit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Splitting the data: 80% training data, 20% testing data
train_data, test_data, train_labels, test_labels = train_test_split(
    all_tweets, labels, test_size=0.2, random_state=1
)
# Printing the number of training and testing samples
print(len(train_data))
print(len(test_data))

# Importing CountVectorizer for converting text data to a bag-of-words representation
from sklearn.feature_extraction.text import CountVectorizer
# Initializing the vectorizer
counter = CountVectorizer()
# Fitting the vectorizer on the training data
counter.fit(train_data)
# Transforming the training and testing data into count matrices
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

# Printing a sample training tweet and its corresponding count vector
print(train_data[3], train_counts[3])

# Importing Multinomial Naive Bayes classifier from scikit-learn
from sklearn.naive_bayes import MultinomialNB
# Initializing the classifier
classifier = MultinomialNB()
# Training the classifier on the count matrix and labels
classifier.fit(train_counts, train_labels)
# Making predictions on the testing data
predictions = classifier.predict(test_counts)

# Importing accuracy_score and confusion_matrix for model evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Printing the accuracy of the model on the testing set
print(accuracy_score(test_labels, predictions))
# Printing the confusion matrix to evaluate classification performance
print(confusion_matrix(test_labels, predictions))

# Example: Predicting the class of a new tweet
tweet = "Hello world!"
# Transforming the tweet into the count matrix format
tweet_counts = counter.transform([tweet])
# Printing the predicted class for the tweet
print(classifier.predict(tweet_counts))
