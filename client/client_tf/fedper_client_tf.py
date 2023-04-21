import numpy as np
import tensorflow as tf

from client.client_tf.client_base_tf import ClientBaseTf
from pathlib import Path
import json
import sys

import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.random.set_seed(0)
class FedPerClientTf(ClientBaseTf):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 epochs					= 1,
				 model_name				= 'DNN',
				 client_selection		= False,
				 solution_name			= 'None',
				 aggregation_method		= 'None',
				 dataset				= '',
				 perc_of_clients		= 0,
				 decay					= 0,
				 fraction_fit			= 0,
				 non_iid				= False,
				 n_personalized_layers	= 1
				 ):

		super().__init__(cid=cid,
						 n_clients=n_clients,
						 n_classes=n_classes,
						 epochs=epochs,
						 model_name=model_name,
						 client_selection=client_selection,
						 solution_name=solution_name,
						 aggregation_method=aggregation_method,
						 dataset=dataset,
						 perc_of_clients=perc_of_clients,
						 decay=decay,
						 fraction_fit=fraction_fit,
						 non_iid=non_iid)

		self.n_personalized_layers = n_personalized_layers*2

	def save_parameters(self):
		try:
			filename = """./fedper_saved_weights/{}/{}/{}.json""".format(self.model_name, self.cid, self.cid)
			weights = self.model.get_weights()
			personalized_layers_weights = []
			for i in range(self.n_personalized_layers):
				personalized_layers_weights.append(weights[len(weights)-self.n_personalized_layers+i])
			data = json.dumps([i.tolist() for i in personalized_layers_weights])
			jsonFile = open(filename, "w")
			jsonFile.write(data)
			jsonFile.close()
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


	def get_parameters(self, config):
		weights = self.model.get_weights()
		self.save_parameters()
		return weights

	def get_parameters_of_model(self):
		weights = self.model.get_weights()
		self.save_parameters()
		return weights

	def set_parameters_to_model(self, parameters):
		try:
			filename = """./fedper_saved_weights/{}/{}/{}.json""".format(self.model_name, self.cid, self.cid)
			if Path(filename).exists():
				fileObject = open(filename, "r")
				jsonContent = fileObject.read()
				aList = [np.array(i) for i in json.loads(jsonContent)]
				size = len(parameters)
				# updating only the personalized layers, which were previously saved in a file
				for i in range(self.n_personalized_layers):
					parameters[size-self.n_personalized_layers+i] = aList[i]
			self.model.set_weights(parameters)
		except Exception as e:
			print("set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

