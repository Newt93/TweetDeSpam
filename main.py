import tweepy
import pandas as pd
import preprocessor as p
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Insert your Twitter API keys here
consumer_key = "YOUR_CONSUMER_KEY"
consumer_secret = "YOUR_CONSUMER_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

# Authenticate with the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create an API object
api = tweepy.API(auth)

# Retrieve tweets containing a specific keyword
tweets = api.search(q="keyword", lang="en", count=100)

# Convert the tweets to a list of strings
tweet_texts = [tweet.text for tweet in tweets]

# Preprocess the data by cleaning it
cleaned_tweet_texts = []
for tweet in tweet_texts:
    cleaned_tweet = p.clean(tweet)
    cleaned_tweet_texts.append(cleaned_tweet)

# Convert the text to a numerical format using TfidfVectorizer
vectorizer = TfidfVectorizer()
tweet_vectors = vectorizer.fit_transform(cleaned_tweet_texts)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tweet_vectors, labels, test_size=0.2)

# Train the neural network using the training set
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate the neural network on the testing set
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Serialize the model and save it to disk
with open('spam_detection_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Use the trained model to classify new tweets as spam or not spam
new_tweet = "This is a new tweet"
cleaned_new_tweet = p.clean(new_tweet)
new_tweet_vector = vectorizer.transform([cleaned_new_tweet])
new_tweet_label = classifier.predict(new_tweet_vector)
print("New tweet label:", new_tweet_label)

# Count the number of tweets that were detected as spam
spam_count = sum(y_pred)

# Count the number of tweets that were not detected as spam
not_spam_count = len(y_pred) - spam_count

# Create a bar chart
plt.bar(['Spam', 'Not Spam'], [spam_count, not_spam_count])
plt.ylabel('Number of Tweets')
plt.show()

# Save the bar chart to a file
plt.savefig('spam_count.png')


