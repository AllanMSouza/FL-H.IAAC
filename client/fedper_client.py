from client.client import Client


import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class FedPerClient(Client):

	def __init__(self, cid,
				 n_clients,
				 epochs=1,
				 model_name         = 'None',
				 client_selection   = False,
				 solution_name      = 'None',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 non_iid            = False):

		self.base_model = None

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

		def create_model(self):
			model, base_model = self._base_create_model()
			self.model = model
			self.base_model = base_model

		def get_parameters(self):
			return self.base_model

		def set_parameters(self, parameters):
			self.base_model.set_weights(parameters)
