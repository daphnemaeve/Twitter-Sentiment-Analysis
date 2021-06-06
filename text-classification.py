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

#pull data from csv file
training = tf.data.experimental.make_csv_dataset(
    "data/training.csv"
    , label_name="label"
    , header=True
    , batch_size=32
    , shuffle_seed=42)

testing = tf.data.experimental.make_csv_dataset(
    "data/testing.csv"
    , label_name="label"
    , header=True
    , batch_size=32
    , shuffle_seed=42)
#print a single batch
iterator = training.as_numpy_iterator()

#select BERT model and preprocessor
bert = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
preprocessor = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

#load preprocessor into hub.KerasLayer
bert_preprocess_model = hub.KerasLayer(preprocessor)

text_test = ['this is a democratic tweet']
text_preprocessed = bert_preprocess_model(text_test)

#load bert model
bert_model = hub.KerasLayer(bert)

bert_results = bert_model(text_preprocessed)

#define bert model
def build_classifier_model():
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(preprocessor, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(bert, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(4, activation=None, name='classifier')(net)
        return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(text_test))

#loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy()
# metrics = tf.metrics.SparseTopKCategoricalAccuracy(k=4)

#optimizer
epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(training).numpy()
print(steps_per_epoch)
num_train_steps = steps_per_epoch * epochs
print(num_train_steps)
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

#load model and train
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=['accuracy'])

print(f'Training model with {bert}')
#also can add validation dataset
history = classifier_model.fit(x=training, epochs=epochs)

dataset_name = 'twitter'
saved_model_path = '/{}_bert'.format(dataset_name.replace('/', '_'))

classifier_model.save(saved_model_path, include_optimizer=False)

#def print_results(inputs, results):
#		result_for_printing = \
#								[f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
#										for i in range(len(inputs))]
#		print(*result_for_printing, sep='\n')
#		print()

#results = classifier_model.predict(testing, verbose=1)

#print('Results:')
#print_results(testing, results)
