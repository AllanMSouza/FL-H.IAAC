import tensorflow as tf
import numpy as np
import random

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

classes_2b_selected = random.randint(2, 5)

for selected_class in range(classes_2b_selected):
	label = random.randint(0, 10)
	
	index_labels = np.where(x_train == label)
	print(y_train[index_labels])