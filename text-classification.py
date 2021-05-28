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
bert_model = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
preprocessor = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

#Load preprocessor into hub.KerasLayer
bert_preprocess_model = hub.KerasLayer(preprocessor)

text_test = ['this is a democratic tweet']
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys	   : {list(text_preprocessed.keys())}')
print(f'Shape	   : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')
