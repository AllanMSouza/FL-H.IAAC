import flwr as fl
import tensorflow
import random
import time
import numpy as np
import tensorflow as tf
import os
import time
import sys

from dataset_utils import ManageDatasets
from model_definition import ModelCreation


import warnings
warnings.simplefilter("ignore")

class FedClient(fl.client.NumPyClient):

	def __init__(self, cid, n_clients, epochs=1, 
				 model_name='None', 
				 client_selection=False, 
				 solution_name='None', 
				 aggregation_method='None',
				 dataset='',
				 perc_of_clients=0,
				 decay=0):

		self.cid          = int(cid)
		self.n_clients    = n_clients
		self.model_name   = model_name
		self.local_epochs = epochs

		self.model        = None
		self.x_train      = None
		self.x_test       = None
		self.y_train      = None
		self.y_test       = None

		#logs
		self.solution_name      = solution_name
		self.aggregation_method = aggregation_method
		self.dataset            = dataset

		self.client_selection = client_selection
		self.perc_of_clients  = perc_of_clients
		self.decay            = decay

		#params
		if self.aggregation_method == 'POC':
			self.solution_name = f"{solution_name}-{aggregation_method}-{self.perc_of_clients}"

		elif self.aggregation_method == 'FedLTA': 
			self.solution_name = f"{solution_name}-{aggregation_method}-{self.decay}"

		elif self.aggregation_method == 'None':
			self.solution_name = f"{solution_name}-{aggregation_method}"

		self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(self.dataset, n_clients=self.n_clients)
		self.model                                           = self.create_model()

	def load_data(self, dataset_name, n_clients):
		return ManageDatasets(self.cid).select_dataset(dataset_name, n_clients)

	def create_model(self):
		input_shape = self.x_train.shape

		if self.model_name == 'Logist Regression':
			return ModelCreation().create_LogisticRegression(input_shape, 10)

		elif self.model_name == 'DNN':
			return ModelCreation().create_DNN(input_shape, 10)

		elif self.model_name == 'CNN':
			return ModelCreation().create_CNN(input_shape, 10)
		

	def get_parameters(self, config):
		return self.model.get_weights()



	def fit(self, parameters, config):
		selected_clients   = []
		trained_parameters = []
		selected           = 0

		if config['selected_clients'] != '':
			selected_clients = [int (cid_selected) for cid_selected in config['selected_clients'].split(' ')]
		
		start_time = time.time()
		print(config)
		if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
			self.model.set_weights(parameters)

			selected           = 1
			history            = self.model.fit(self.x_train, self.y_train, verbose=1, epochs=self.local_epochs)
			trained_parameters = self.model.get_weights()
		
		total_time         = time.time() - start_time
		size_of_parameters = sum(map(sys.getsizeof, trained_parameters))
		avg_loss_train     = np.mean(history.history['loss'])
		avg_acc_train      = np.mean(history.history['accuracy'])

		filename = f"logs/{self.solution_name}/{self.n_clients}/{self.model_name}/{self.dataset}/train_client.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)

		with open(filename, 'a') as log_train_file:
			log_train_file.write(f"{config['round']}, {self.cid}, {selected}, {total_time}, {size_of_parameters}, {avg_loss_train}, {avg_acc_train}\n")

		fit_response = {
			'cid' : self.cid
		}

		return trained_parameters, len(self.x_train), fit_response


	def evaluate(self, parameters, config):
		
		self.model.set_weights(parameters)
		loss, accuracy     = self.model.evaluate(self.x_test, self.y_test, verbose=1)
		size_of_parameters = sum(map(sys.getsizeof, parameters))

		filename = f"logs/{self.solution_name}/{self.n_clients}/{self.model_name}/{self.dataset}/evaluate_client.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)

		with open(filename, 'a') as log_evaluate_file:
			log_evaluate_file.write(f"{config['round']}, {self.cid}, {size_of_parameters}, {loss}, {accuracy}\n")

		evaluation_response = {
			"cid"      : self.cid,
			"accuracy" : float(accuracy)
		}

		return loss, len(self.x_test), evaluation_response


