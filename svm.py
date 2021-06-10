#!/usr/bin/env python

import os
import shutil

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

#pull data from csv file
training = pd.read_csv("data/training.csv")
testing = pd.read_csv("data/testing.csv")
print(training.head())

training_text = training['text']
training_labels = training['label']
testing_text = testing['text']
testing_labels = testing['label']

count_vect = CountVectorizer()
training_counts = count_vect.fit_transform(training_text)

tfidf_transformer = TfidfTransformer(use_idf=False).fit(training_counts)
training_tfidf = tfidf_transformer.transform(training_counts)

#clf = svm.SVC()
#clf.fit(training_tfidf, training_labels)

parameters = {
				'vect__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
				'tfidf__use_idf': (True, False),
				'clf__alpha': (1e-2, 1e-3),
}

#clf = MultinomialNB().fit(training_tfidf, training_labels)

text_clf = Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', SGDClassifier(loss='hinge', penalty='l2',
							  alpha=1e-3, random_state=42,
							  max_iter=5, tol=None)),
		])

text_clf.fit(training_text, training_labels)

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(training_text, training_labels)

print(gs_clf.best_score_)
for param_name in sorted(parameters.keys()):
		print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

predicted = gs_clf.predict(testing_text)
for i in range(50):
	print("Text: {}".format(testing_text[i]))
	print("Predicted party: {}".format(predicted[i]))
	print("Actual party: {}".format(testing_labels[i]))

print("Accuracy: ")
print(np.mean(predicted == testing_labels))
