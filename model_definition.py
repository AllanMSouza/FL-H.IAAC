import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, Flatten, MaxPool2D, Dense, InputLayer, BatchNormalization, Dropout

#from sklearn.linear_model import LogisticRegression
import numpy as np

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class ModelCreation():

	def create_DNN(self, input_shape, num_classes):
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Flatten(input_shape=(input_shape[1:])))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(64,  activation='relu'))
		model.add(Dense(32,  activation='relu'))
		model.add(Dense(num_classes, activation='softmax'))

		model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

		return model


	def create_CNN(self, input_shape, num_classes):

		# deep_cnn = Sequential()

		# if len(input_shape) == 3:
		# 	deep_cnn.add(InputLayer(input_shape=(input_shape[1], input_shape[2], 1)))
		# else:
		# 	deep_cnn.add(InputLayer(input_shape=(input_shape[1:])))

		# deep_cnn.add(Conv2D(128, (5, 5), activation='relu', strides=(1, 1), padding='same'))
		# deep_cnn.add(MaxPool2D(pool_size=(2, 2)))

		# deep_cnn.add(Conv2D(64, (5, 5), activation='relu', strides=(2, 2), padding='same'))
		# deep_cnn.add(MaxPool2D(pool_size=(2, 2)))
		# deep_cnn.add(BatchNormalization())

		# deep_cnn.add(Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same'))
		# deep_cnn.add(MaxPool2D(pool_size=(2, 2)))
		# deep_cnn.add(BatchNormalization())

		# deep_cnn.add(Flatten())

		# deep_cnn.add(Dense(100, activation='relu'))
		# deep_cnn.add(Dense(100, activation='relu'))
		# deep_cnn.add(Dropout(0.25))

		# deep_cnn.add(Dense(num_classes, activation='softmax'))

		# deep_cnn.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

		# return deep_cnn
		
		deep_cnn = Sequential()
		deep_cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu',kernel_initializer='he_uniform', input_shape=(input_shape[1], 1)))
		deep_cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu',kernel_initializer='he_uniform'))
		deep_cnn.add(Dropout(0.6))
		deep_cnn.add(MaxPooling1D(pool_size=2))
		deep_cnn.add(Flatten())
		deep_cnn.add(Dense(50, activation='relu'))
		deep_cnn.add(Dense(num_classes, activation='softmax'))
	
		deep_cnn.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

		return deep_cnn


	def create_LogisticRegression(self, input_shape, num_classes):

		logistic_regression = Sequential()

		if len(input_shape) == 3:
			logistic_regression.add(Flatten(input_shape=(input_shape[1], input_shape[2], 1)))
		else:
			logistic_regression.add(Flatten(input_shape=(input_shape[1:])))

		logistic_regression.add(Dense(num_classes, activation='sigmoid'))
		logistic_regression.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

		return logistic_regression





