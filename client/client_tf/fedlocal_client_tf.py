import numpy as np
import tensorflow as tf

from client.client_tf.client_base_tf import ClientBaseTf
from pathlib import Path
import json

import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.random.set_seed(0)
class FedLocalClientTf(ClientBaseTf):

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
						 non_iid=non_iid)

		self.n_personalized_layers = n_personalized_layers*2

	def save_parameters(self):
		filename = """./fedlocal_saved_weights/{}/{}/{}.json""".format(self.model_name, self.cid, self.cid)
		weights = self.model.get_weights()
		data = json.dumps([i.tolist() for i in weights])
		jsonFile = open(filename, "w")
		jsonFile.write(data)
		jsonFile.close()

	def get_parameters(self, config):
		weights = self.model.get_weights()
		self.save_parameters()
		return weights

	def get_parameters_of_model(self):
		weights = self.model.get_weights()
		self.save_parameters()
		return weights

	def set_parameters_to_model(self, parameters):
		filename = """./fedlocal_saved_weights/{}/{}/{}.json""".format(self.model_name, self.cid, self.cid)
		if Path(filename).exists():
			fileObject = open(filename, "r")
			jsonContent = fileObject.read()
			aList = [np.array(i) for i in json.loads(jsonContent)]
			parameters = aList
		self.model.set_weights(parameters)
