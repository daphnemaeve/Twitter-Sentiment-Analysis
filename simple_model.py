#!/usr/bin/env python

import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np
from official.nlp import optimization
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

#pull data from csv file
training = pd.read_csv("data/training.csv")
testing = pd.read_csv("data/testing.csv")
print(training.head())

training_text = training['text']
training_labels = training['label']

count_vect = CountVectorizer()
training_counts = count_vect.fit_transform(training_text)

tfidf_transformer = TfidfTransformer(use_idf=False).fit(training_counts)
training_tfidf = tfidf_transformer.transform(training_counts)

clf = MultinomialNB().fit(training_tfidf, training_labels)

test_tweets = ['The United States just hit 50,000 new coronavirus cases in a day yesterday—and we were warned we could see 100,000 per day. Instead of taking meaningful action to reduce cases, Trump is waving the white flag.','McConnell, Pence, Barr, Pompeo, Miller... they all got away with it. Some already being rehabilitated by the ""resistance.""','The elections are no joking matter.','This is what real election interference looks like.']

new_counts = count_vect.transform(test_tweets)
new_tfidf = tfidf_transformer.transform(new_counts)

predicted = clf.predict(new_tfidf)
print(predicted)

for i in range(4):
	print(test_tweets[i] + " => {}".format(predicted[i]))

text_clf = Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', MultinomialNB()),
		])

text_clf.fit(training_text, training_labels)

testing_text = testing['text']
testing_labels = testing['label']

predicted = text_clf.predict(testing_text)
print(np.mean(predicted == testing_labels))
