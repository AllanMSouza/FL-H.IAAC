from client.client_torch.client_base_torch import ClientBaseTorch
from torch.nn.parameter import Parameter
import torch
import json
from pathlib import Path
import numpy as np
import os
import sys

import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class FedLocalClientTorch(ClientBaseTorch):

	def __init__(self,
                 cid,
                 n_clients,
                 n_classes,
				 args,
                 epochs=1,
                 model_name         = 'DNN',
                 client_selection   = False,
                 strategy_name      ='FedLocal',
                 aggregation_method = 'None',
                 dataset            = '',
                 perc_of_clients    = 0,
                 decay              = 0,
				 fraction_fit		= 0,
                 non_iid            = False,
                 n_personalized_layers	= 1,
                 new_clients			= False,
                 new_clients_train	= False
                 ):

		super().__init__(cid=cid,
                         n_clients=n_clients,
                         n_classes=n_classes,
                         epochs=epochs,
                         model_name=model_name,
                         client_selection=client_selection,
                         solution_name=strategy_name,
                         aggregation_method=aggregation_method,
                         dataset=dataset,
                         perc_of_clients=perc_of_clients,
                         decay=decay,
						 fraction_fit=fraction_fit,
                         non_iid=non_iid,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train,
						 args=args)

		self.n_personalized_layers = n_personalized_layers * 2

	def set_parameters_to_model(self, parameters):
		try:
			filename = """./fedlocal_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid, self.cid)
			if os.path.exists(filename):
				self.model.load_state_dict(torch.load(filename))
			else:
				# parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
				# for new_param, old_param in zip(parameters, self.model.parameters()):
				# 	old_param.data = new_param.data.clone()
				pass
		except Exception as e:
			print("load and set parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_parameters(self):
		# os.makedirs("""{}/fedproto_saved_weights/{}/{}/""".format(os.getcwd(), self.model_name, self.cid),
		# 			exist_ok=True)
		try:
			filename = """./fedlocal_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid)
			if Path(filename).exists():
				os.remove(filename)
			torch.save(self.model.state_dict(), filename)
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
