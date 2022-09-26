import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Dense, InputLayer, BatchNormalization, Dropout

from sklearn.linear_model import LogisticRegression

class ModelCreation():

	def create_DNN(self, input_shape, num_classes):
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Flatten(input_shape=(input_shape[1:])))
		model.add(Dense(512, activation='relu'))
		model.add(Dense(256, activation='relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(num_classes, activation='softmax'))

		model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

		return model


	def create_CNN(self, input_shape, num_classes):

		deep_cnn = Sequential()

		if len(input_shape) == 3:
			deep_cnn.add(InputLayer(input_shape=(input_shape[1], input_shape[2], 1)))
		else:
			deep_cnn.add(InputLayer(input_shape=(input_shape[1:])))

		deep_cnn.add(Conv2D(128, (5, 5), activation='relu', strides=(1, 1), padding='same'))
		deep_cnn.add(MaxPool2D(pool_size=(2, 2)))

		deep_cnn.add(Conv2D(64, (5, 5), activation='relu', strides=(2, 2), padding='same'))
		deep_cnn.add(MaxPool2D(pool_size=(2, 2)))
		deep_cnn.add(BatchNormalization())

		deep_cnn.add(Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same'))
		deep_cnn.add(MaxPool2D(pool_size=(2, 2)))
		deep_cnn.add(BatchNormalization())

		deep_cnn.add(Flatten())

		deep_cnn.add(Dense(100, activation='relu'))
		deep_cnn.add(Dense(100, activation='relu'))
		deep_cnn.add(Dropout(0.25))

		deep_cnn.add(Dense(num_classes, activation='softmax'))

		deep_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

		return deep_cnn


	def create_MobileNet(self, input_shape, num_classes):

		if len(input_shape) == 3:
			model = tf.keras.applications.MobileNetV2(input_shape=(input_shape[1], input_shape[2], 1), classes=num_classes, weights=None, alpha=0.1) 
		else:
			model = tf.keras.applications.MobileNetV2(input_shape=(input_shape[1:]), classes=num_classes, weights=None) 
		
		model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

		return model





