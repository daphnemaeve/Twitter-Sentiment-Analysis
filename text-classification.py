#!/usr/bin/env python

import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

training = tf.data.experimental.make_csv_dataset(
    "data/training.csv"
    , label_name="party"
    , header=True
    , batch_size=32
    , shuffle_seed=42)

#print a single batch
iterator = training.as_numpy_iterator()
print(next(iterator))

#Select BERT model and preprocessor
bert = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
preprocessor = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

#Load preprocessor into hub.KerasLayer
bert_preprocess_model = hub.KerasLayer(preprocessor)

text_test = ['this is a democratic tweet']
text_preprocessed = bert_preprocess_model(text_test)

#print(f'Keys	   : {list(text_preprocessed.keys())}')
#print(f'Shape	   : {text_preprocessed["input_word_ids"].shape}')
#print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
#print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
#print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

bert_model = hub.KerasLayer(bert)

bert_results = bert_model(text_preprocessed)

#print(f'Loaded BERT: {bert}')
#print(f'Pooled Outputs Shape: {bert_results["pooled_output"].shape}')
#print(f'Pooled Outputs Values: {bert_results["pooled_output"][0, :12]}')
#print(f'Sequence Outputs Shape: {bert_results["sequence_output"].shape}')
#print(f'Sequence Outputs Values: {bert_results["sequence_output"][0, :12]}')

def build_classifier_model():
	text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
	preprocessing_layer = hub.KerasLayer(preprocessor, name='preprocessing')
	encoder_inputs = preprocessing_layer(text_input)
	encoder = hub.KerasLayer(bert, trainable=True, name='BERT_encoder')
	outputs = encoder(encoder_inputs)
	net = outputs['pooled_output']
	net = tf.keras.layers.Dropout(0.1)(net)
	net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
	return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))
