from client.client_torch import FedAvgClientTorch
from client.client_torch.fedper_client_torch import FedPerClientTorch
from ..fedpredict_core import fedpredict_core
from torch.nn.parameter import Parameter
import torch
import json
import sys
import time
import copy
import torch
import torch.nn as nn
import torch
from torch.optim import Optimizer
from client.client_torch.client_base_torch import ClientBaseTorch
# from ...optimizers import PerturbedGradientDescent


import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)



class PerturbedGradientDescent(Optimizer):
	def __init__(self, params, lr=0.01, mu=0.0):
		default = dict(lr=lr, mu=mu)
		super().__init__(params, default)

	@torch.no_grad()
	def step(self, global_params, device):
		try:
			for group in self.param_groups:
				for p, g in zip(group['params'], global_params):
					g = torch.Tensor(g).to(device)
					d_p = p.grad.data + group['mu'] * (p.data - g.data)
					p.data.add_(d_p, alpha=-group['lr'])
					# print("ola: ", group['mu'], group['lr'])
		except Exception as e:
			print("pertubed gradient descent")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

class FedProxClientTorch(FedAvgClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedProx',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 fraction_fit		= 0,
				 non_iid            = False,
				 m_combining_layers	= 1,
				 new_clients			= False,
				 new_clients_train	= False
				 ):

		super().__init__(cid=cid,
						 n_clients=n_clients,
						 n_classes=n_classes,
						 args=args,
						 epochs=epochs,
						 model_name=model_name,
						 client_selection=client_selection,
						 strategy_name=strategy_name,
						 aggregation_method=aggregation_method,
						 dataset=dataset,
						 perc_of_clients=perc_of_clients,
						 decay=decay,
						 fraction_fit=fraction_fit,
						 non_iid=non_iid,
						 new_clients=new_clients,
						 new_clients_train=new_clients_train)

		self.mu = 0.1
		self.learning_rate_decay_gamma = 0.9
		self.global_params = copy.deepcopy(list(self.model.parameters()))
		self.loss = nn.CrossEntropyLoss()
		self.optimizer = PerturbedGradientDescent(
			self.model.parameters(), lr=self.learning_rate, mu=self.mu)
		self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
			optimizer=self.optimizer,
			gamma=self.learning_rate_decay_gamma
		)

	# def set_parameters_to_model_fit(self, parameters):
	# 	for new_param, global_param, param in zip(parameters, self.global_params, self.model.parameters()):
	# 		global_param.data = new_param.data.clone()
	# 		param.data = new_param.data.clone()

	def fit(self, parameters, config):
		try:
			selected_clients = []
			trained_parameters = []
			selected = 0

			if config['selected_clients'] != '':
				selected_clients = [int(cid_selected) for cid_selected in config['selected_clients'].split(' ')]

			start_time = time.process_time()
			server_round = int(config['round'])
			if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
				self.set_parameters_to_model_fit(parameters)
				self.round_of_last_fit = server_round

				selected = 1
				self.model.train()

				max_local_steps = self.local_epochs
				train_acc = 0
				train_loss = 0
				train_num = 0
				for step in range(max_local_steps):
					for i, (x, y) in enumerate(self.trainloader):
						if type(x) == type([]):
							x[0] = x[0].to(self.device)
						else:
							x = x.to(self.device)
						y = y.to(self.device)
						train_num += y.shape[0]

						self.optimizer.zero_grad()
						output = self.model(x)
						y = torch.tensor(y.int().detach().numpy().astype(int).tolist())
						loss = self.loss(output, y)
						train_loss += loss.item() * y.shape[0]
						loss.backward()
						self.optimizer.step(parameters, self.device)

						train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

				trained_parameters = self.get_parameters_of_model()
				self.save_parameters()


			size_of_parameters = sum(
				[sum(map(sys.getsizeof, trained_parameters[i])) for i in range(len(trained_parameters))])
			avg_loss_train = train_loss / train_num
			avg_acc_train = train_acc / train_num
			total_time = time.process_time() - start_time
			# loss, accuracy, test_num = self.model_eval()

			data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]

			self._write_output(
				filename=self.train_client_filename,
				data=data)

			fit_response = {
				'cid': self.cid
			}

			return trained_parameters, train_num, fit_response
		except Exception as e:
			print("fit")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)