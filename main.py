#Library imports

import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download required NLTK data
nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Load positive, negative and neutral tweets data
pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')
neutral_tweets = twitter_samples.strings('tweets.20150430-223406.json')

# Tokenize, remove stopwords and stem words for all tweets
stop_words = stopwords.words('english')
stemmer = PorterStemmer()

def clean_tweet(tweet):
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.\S+', '', tweet)
    # Remove Twitter mentions
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(tweet.lower())
    # Remove stopwords and stem words
    words = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(words)

pos_tweets_cleaned = [clean_tweet(tweet) for tweet in pos_tweets]
neg_tweets_cleaned = [clean_tweet(tweet) for tweet in neg_tweets]
neutral_tweets_cleaned = [clean_tweet(tweet) for tweet in neutral_tweets]

# Combine cleaned tweets with their sentiment labels
tweets = pos_tweets_cleaned + neg_tweets_cleaned + neutral_tweets_cleaned
labels = ['positive']*len(pos_tweets_cleaned) + ['negative']*len(neg_tweets_cleaned) + ['neutral']*len(neutral_tweets_cleaned)

# Train logistic regression model with gradient descent and sigmoid function
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(tweets)
y = labels
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X, y)

# Setting Title of App
st.title("Twitter Sentiment Analysis")

# Get user input
tweet = st.text_input("Enter a tweet:")
st.write("Example - Negative Sentiment: I just wasted two hours of my life watching that movie. It was the worst thing I've ever seen.")
st.write("Example - Positive Sentiment: I had an amazing day at the beach with my friends!")

# On predict button click
if st.button('Predict'):
    if tweet:
        # Clean and vectorize the user input tweet
        tweet_cleaned = clean_tweet(tweet)
        tweet_vec = tfidf.transform([tweet_cleaned])

        # Make prediction and display sentiment
        pred = lr.predict(tweet_vec)[0]
        if pred == 'positive':
            st.write("Sentiment: Positive")
        elif pred == 'negative':
            st.write("Sentiment: Negative")
        else:
            st.write("Sentiment: Neutral")
    else:
        st.write("Please enter a tweet to predict its sentiment.")

