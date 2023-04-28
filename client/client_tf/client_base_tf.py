import flwr as fl
import tensorflow
import random
import time
import numpy as np
import tensorflow as tf
import os
import time
import sys

from dataset_utils_tf import ManageDatasets
from model_definition_tf import ModelCreation
import csv


import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class ClientBaseTf(fl.client.NumPyClient):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs				= 1,
				 model_name         = 'None',
				 client_selection   = False,
				 solution_name      = 'None',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 fraction_fit		= 0,
				 decay              = 0,
				 non_iid            = False,
				 new_clients = False,
				 new_clients_train	= False
				 ):

		self.cid          = int(cid)
		self.n_clients    = n_clients
		self.model_name   = model_name
		self.local_epochs = epochs
		self.non_iid      = non_iid
		# self.n_rounds	  = int(args.round)

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
		self.fraction_fit = fraction_fit

		self.loss = tf.keras.losses.CategoricalCrossentropy
		self.learning_rate = 0.01
		self.new_clients = new_clients
		self.new_clients_train = new_clients_train
		# self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.device = tf.device("cuda:0" if tf.test.is_gpu_available() else "cpu")
		self.type = 'tf'

		#params
		if self.aggregation_method == 'POC':
			self.solution_name = f"{solution_name}-{aggregation_method}-{self.perc_of_clients}"

		elif self.aggregation_method == 'FedLTA': 
			self.solution_name = f"{solution_name}-{aggregation_method}-{self.decay}"

		elif self.aggregation_method == 'None':
			self.solution_name = f"{solution_name}-{aggregation_method}-{self.fraction_fit}"

		self.base = f"logs/{self.type}/{self.solution_name}/new_clients_{self.new_clients}_train_{self.new_clients_train}/{self.n_clients}/{self.model_name}/{self.dataset}/{self.n_rounds}_rounds/{self.local_epochs}_local_epochs"
		self.evaluate_client_filename = f"{self.base}/evaluate_client.csv"
		self.train_client_filename = f"{self.base}/train_client.csv"

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
		try:
			selected_clients   = []
			trained_parameters = []
			selected           = 0

			if config['selected_clients'] != '':
				selected_clients = [int (cid_selected) for cid_selected in config['selected_clients'].split(' ')]

			start_time = time.process_time()
			if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
				self.set_parameters_to_model(parameters)

				selected           = 1
				history            = self.model.fit(self.x_train, self.y_train, verbose=0, epochs=self.local_epochs)
				trained_parameters = self.get_parameters_of_model()

			total_time         = time.process_time() - start_time
			size_of_parameters = sum(map(sys.getsizeof, trained_parameters))
			avg_loss_train     = np.mean(history.history['loss'])
			avg_acc_train      = np.mean(history.history['accuracy'])

			data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]

			self._write_output(
				filename=self.train_client_filename,
				data=data)

			fit_response = {
				'cid' : self.cid
			}

			return trained_parameters, len(self.x_train), fit_response

		except Exception as e:
			print("fit")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)



	def evaluate(self, parameters, config):
		try:
			self.set_parameters_to_model(parameters)
			loss, accuracy     = self.model.evaluate(self.x_test, self.y_test, verbose=0)
			size_of_parameters = sum(map(sys.getsizeof, parameters))

			data = [config['round'], self.cid, size_of_parameters, loss, accuracy]

			self._write_output(filename=self.evaluate_client_filename,
							   data=data)

			evaluation_response = {
				"cid"      : self.cid,
				"accuracy" : float(accuracy)
			}

			return loss, len(self.x_test), evaluation_response

		except Exception as e:
			print("evaluate")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


	def _write_output(self, filename, data):
		try:
			with open(filename, 'a') as server_log_file:
				writer = csv.writer(server_log_file)
				writer.writerow(data)
		except Exception as e:
			print("write output")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)



