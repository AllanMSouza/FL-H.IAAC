from client.client import Client


import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class FedPerClient(Client):

	def __init__(self, cid,
				 n_clients,
				 epochs=1,
				 model_name         = 'DNN_transfer_learning',
				 client_selection   = False,
				 solution_name      = 'None',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 non_iid            = False):

		super().__init__(cid=cid,
				 n_clients=n_clients,
				 epochs=epochs,
				 model_name=model_name,
				 client_selection=client_selection,
				 solution_name=solution_name,
				 aggregation_method=aggregation_method,
				 dataset=dataset,
				 perc_of_clients=perc_of_clients,
				 decay=decay,
				 non_iid=non_iid)

	def get_parameters(self, config):
		return self.model.base_model.get_weights()

	def get_parameters_of_model(self):
		return self.model.base_model.get_weights()

	def set_parameters_to_model(self, parameters):
		self.model.base_model.set_weights(parameters)
