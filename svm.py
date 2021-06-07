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
from sklearn.linear_model import SGDClassifier
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

test_tweets = ['The United States just hit 50,000 new coronavirus cases in a day yesterday—and we were warned we could see 100,000 per day. Instead of taking meaningful action to reduce cases, Trump is waving the white flag.','McConnell, Pence, Barr, Pompeo, Miller... they all got away with it. Some already being rehabilitated by the ""resistance.""','The elections are no joking matter.','This is what real election interference looks like.']

new_counts = count_vect.transform(test_tweets)
new_tfidf = tfidf_transformer.transform(new_counts)

#clf = svm.SVC()
#clf.fit(training_tfidf, training_labels)

clf = MultinomialNB().fit(training_tfidf, training_labels)

text_clf = Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', SGDClassifier(loss='hinge', penalty='l2',
							  alpha=1e-3, random_state=42,
							  max_iter=5, tol=None)),
		])

text_clf.fit(training_text, training_labels)

testing_text = testing['text']
testing_labels = testing['label']

predicted = text_clf.predict(testing_text)
for i in range(50):
	print("Text: {}".format(testing_text[i]))
	print("Predicted party: {}".format(predicted[i]))
	print("Actual party: {}".format(testing_labels[i]))

print("Accuracy: ")
print(np.mean(predicted == testing_labels))
