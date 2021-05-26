import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

dataset = open("data/labeled_data.csv")

#need to split data into training and testing

print(dataset)