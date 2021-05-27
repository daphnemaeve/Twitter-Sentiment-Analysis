#!/usr/bin/env python

import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

dataset = tf.data.experimental.make_csv_dataset(
    "data/training.csv"
    , label_name="party"
    , header=True
    , batch_size=32
    , shuffle_seed=42)

#print a single batch
iterator = dataset.as_numpy_iterator()
print(next(iterator))
