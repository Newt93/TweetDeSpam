# TweetDeSpam
A neural network that pulls tweets from the Twitter API and uses a classifier to determine if a tweet is spam or not spam, then visualizes the data with matplotlib

This is a script that connects to the Twitter API using the tweepy library, it then uses the twitter API to collect tweets, it then preprocesses the tweets by cleaning the text, removing stopwords, and converting it to numerical format. After that, it splits the dataset into training and testing sets, trains a neural network using the training set, evaluates the neural network on the testing set, serializes the model and saves it to disk, and using the trained model to classify new tweets as spam or not spam as they come in.
It then uses matplotlib to visualize the data that has been detected as spam or not spam, which includes a bar chart that shows the number of tweets that were detected as spam or not spam, and a confusion matrix that shows the performance of the classifier.
Last, it saves the plotted data in an image file using the savefig() function of matplotlib.

If you'd like a custom machine learning program built please message leakzgggaming@gmail.com
