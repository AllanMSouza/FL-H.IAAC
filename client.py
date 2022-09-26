import flwr as fl
import tensorflow
import random
import time
import numpy as np
import tensorflow as tf


from dataset_utils import ManageDatasets
from model_definition import ModelCreation


import warnings
warnings.simplefilter("ignore")

class FedClient(fl.client.NumPyClient):

	def __init__(self, cid, n_clients, model_name='DNN', client_selection=False):
		self.cid          = int(cid)
		self.n_clients    = n_clients
		self.model_name   = model_name

		self.model        = None
		self.local_epochs = 1
		self.x_train      = None
		self.x_test       = None
		self.y_train      = None
		self.y_test       = None


		self.client_selection = client_selection

		self.x_train, self.y_train, self.x_test, self.y_test = self.load_data('CIFAR10', n_clients=self.n_clients)
		self.model                                           = self.create_model()

	def load_data(self, dataset_name, n_clients):
		return ManageDatasets(self.cid).select_dataset(dataset_name, n_clients)

	def create_model(self):
		input_shape = self.x_train.shape
	
		#return ModelCreation().create_DNN(input_shape, 10)
		#return ModelCreation().create_CNN(input_shape, 100)
		return ModelCreation().create_MobileNet(input_shape, 10)
		

	def get_parameters(self, config):

		#return self.model.get_weights() if self.model_name != 'LogisticRegression' else self.model.get_params()
		return self.model.get_weights()



	def fit(self, parameters, config):
		selected_clients = []
		

		if config['selected_clients'] != '':
			selected_clients = [int (cid_selected) for cid_selected in config['selected_clients'].split(' ')]
		
		print(f'CLIENT SELECTED_CLIENTS: {selected_clients}')
		if self.cid in selected_clients or self.client_selection == False:
			self.model.set_weights(parameters)
			self.model.fit(self.x_train, self.y_train, verbose=1, epochs=self.local_epochs)

		fit_response = {
			'cid' : self.cid
		}

		return self.model.get_weights(), len(self.x_train), fit_response


	def evaluate(self, parameters, config):
		self.model.set_weights(parameters)
		loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=1)

		evaluation_response = {
			"cid"      : self.cid,
			"accuracy" : float(accuracy)
		}

		return loss, len(self.x_test), evaluation_response


