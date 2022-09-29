import tensorflow as tf
import numpy as np
import random

from sklearn.preprocessing import Normalizer

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class ManageDatasets():

	def __init__(self, cid):
		self.cid = cid
		random.seed(self.cid)

	def load_MNIST(self, n_clients):

		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
		x_train, x_test                      = x_train/255.0, x_test/255.0
		x_train, y_train, x_test, y_test     = self.slipt_dataset(x_train, y_train, x_test, y_test, n_clients)

		return x_train, y_train, x_test, y_test

	def load_CIFAR10(self, n_clients):
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
		x_train, x_test                      = x_train/255.0, x_test/255.0
		x_train, y_train, x_test, y_test     = self.slipt_dataset(x_train, y_train, x_test, y_test, n_clients)

		return x_train, y_train, x_test, y_test


	def load_CIFAR100(self, n_clients):
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
		x_train, x_test                      = x_train/255.0, x_test/255.0
		x_train, y_train, x_test, y_test     = self.slipt_dataset(x_train, y_train, x_test, y_test, n_clients)

		return x_train, y_train, x_test, y_test


	def slipt_dataset(self, x_train, y_train, x_test, y_test, n_clients):
		p_train = int(len(x_train)/n_clients)
		p_test  = int(len(x_test)/n_clients)


		random.seed(self.cid)
		selected_train = random.sample(range(len(x_train)), p_train)

		random.seed(self.cid)
		selected_test  = random.sample(range(len(x_test)), p_test)
		
		x_train  = x_train[selected_train]
		y_train  = y_train[selected_train]

		x_test   = x_test[selected_test]
		y_test   = y_test[selected_test]


		return x_train, y_train, x_test, y_test


	def select_dataset(self, dataset_name, n_clients):

		if dataset_name == 'MNIST':
			return self.load_MNIST(n_clients)

		elif dataset_name == 'CIFAR100':
			return self.load_CIFAR100(n_clients)

		elif dataset_name == 'CIFAR10':
			return self.load_CIFAR10(n_clients)


	def normalize_data(self, x_train, x_test):
		x_train = Normalizer().fit_transform(np.array(x_train))
		x_test  = Normalizer().fit_transform(np.array(x_test))
		return x_train, x_test


