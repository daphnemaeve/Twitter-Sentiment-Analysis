#!/usr/bin/env python

import os
import shutil

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

#pull data from csv file
training = pd.read_csv("data/large_training.csv")
testing = pd.read_csv("data/large_testing.csv")
print(training.head())

training_text = training['text']
training_labels = training['label']

count_vect = CountVectorizer()
training_counts = count_vect.fit_transform(training_text)

tfidf_transformer = TfidfTransformer(use_idf=False).fit(training_counts)
training_tfidf = tfidf_transformer.transform(training_counts)

clf = MultinomialNB().fit(training_tfidf, training_labels)

text_clf = Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', MultinomialNB()),
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
