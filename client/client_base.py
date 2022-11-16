import flwr as fl
import numpy as np
import tensorflow as tf
import os
import time
import sys

from dataset_utils import ManageDatasets
from model_definition import ModelCreation
import csv


# import warnings
# warnings.simplefilter("ignore")
#
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)

class ClientBase(fl.client.NumPyClient):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 epochs=1,
				 model_name         = 'None',
				 client_selection   = False,
				 solution_name      = 'None',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 non_iid            = False,
				 ):

		self.cid          = int(cid)
		self.n_clients    = n_clients
		self.model_name   = model_name
		self.local_epochs = epochs
		self.non_iid      = non_iid

		self.epochs = epochs
		self.num_classes = n_classes


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
		return ManageDatasets(self.cid).select_dataset(dataset_name, n_clients, self.non_iid)

	def create_model(self):
		input_shape = self.x_train.shape

		if self.model_name == 'Logist Regression':
			return ModelCreation().create_LogisticRegression(input_shape, self.num_classes)

		elif self.model_name == 'DNN':
			return ModelCreation().create_DNN(input_shape, self.num_classes)

		elif self.model_name == 'CNN':
			return ModelCreation().create_CNN(input_shape, self.num_classes)

		else:
			raise Exception("Wrong model name")

	def get_parameters(self, config):
		return self.model.get_weights()

	# It does the same of "get_parameters", but using "get_parameters" in outside of the core of Flower is causing errors
	def get_parameters_of_model(self):
		return self.model.get_weights()

	def set_parameters_to_model(self, parameters):
		return self.model.set_weights(parameters)

	def fit(self, parameters, config):
		selected_clients   = []
		trained_parameters = []
		selected           = 0

		if config['selected_clients'] != '':
			selected_clients = [int (cid_selected) for cid_selected in config['selected_clients'].split(' ')]
		
		start_time = time.process_time()
		#print(config)
		if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
			self.set_parameters_to_model(parameters)

			selected           = 1
			history            = self.model.fit(self.x_train, self.y_train, verbose=0, epochs=self.local_epochs)
			trained_parameters = self.get_parameters_of_model()
		
		total_time         = time.process_time() - start_time
		size_of_parameters = sum(map(sys.getsizeof, trained_parameters))
		avg_loss_train     = np.mean(history.history['loss'])
		avg_acc_train      = np.mean(history.history['accuracy'])

		filename = f"logs/{self.solution_name}/{self.n_clients}/{self.model_name}/{self.dataset}/train_client.csv"
		data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]

		self._write_output(
			filename=filename,
			data=data)

		fit_response = {
			'cid' : self.cid
		}

		return trained_parameters, len(self.x_train), fit_response


	def evaluate(self, parameters, config):
		
		self.set_parameters_to_model(parameters)
		loss, accuracy     = self.model.evaluate(self.x_test, self.y_test, verbose=0)
		size_of_parameters = sum(map(sys.getsizeof, parameters))

		filename = f"logs/{self.solution_name}/{self.n_clients}/{self.model_name}/{self.dataset}/evaluate_client.csv"
		data = [config['round'], self.cid, size_of_parameters, loss, accuracy]

		self._write_output(filename=filename,
						   data=data)

		evaluation_response = {
			"cid"      : self.cid,
			"accuracy" : float(accuracy)
		}

		return loss, len(self.x_test), evaluation_response

	def _write_output(self, filename, data):

		with open(filename, 'a') as server_log_file:
			writer = csv.writer(server_log_file)
			writer.writerow(data)


