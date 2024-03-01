import copy
import os
import time
import sys
from collections import defaultdict
from pathlib import Path
from models.torch import DNN_proto, Logistic_Proto, CNN_proto
import torch.nn as nn
import warnings

warnings.simplefilter("ignore")
from client.client_torch.client_base_torch import ClientBaseTorch
from models.torch import DNN, Logistic, CNN, MobileNet, resnet20, CNN_EMNIST, MobileNetV2, CNN_X, CNN_5, CNN_2, CNN_3_proto
# logging.getLogger("torch").setLevel(logging.ERROR)
from torch.nn.parameter import Parameter
import torch
from sklearn.metrics import f1_score
import numpy as np
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
class FedProtoClientTorch(ClientBaseTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs				= 1,
				 model_name         = 'None',
				 client_selection   = False,
				 strategy_name      ='None',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 fraction_fit		= 0,
				 non_iid            = False,
				 new_clients		= False,
				 new_clients_train	= False
				 ):

		super().__init__(cid=cid,
						 n_clients=n_clients,
						 n_classes=n_classes,
						 args=args,
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
						 new_clients_train=new_clients_train)

		self.protos = None
		self.global_protos = None
		self.loss_mse = nn.MSELoss()

		self.lamda = 1
		self.protos_samples_per_class = {i: 0 for i in range(self.num_classes)}

	def create_model(self):

		try:
			model = None
			if self.dataset in ['MNIST', 'EMNIST']:
				input_shape = 1
			elif self.dataset in ['CIFAR10', 'GTSRB']:
				input_shape = 3
			if self.model_name in ['CNN_3'] and self.dataset in ['EMNIST', 'MNIST', 'CIFAR10', 'GTSRB']:
				if self.dataset in ['CIFAR10']:
					mid_dim = 16
				elif self.dataset == 'GTSRB':
					mid_dim = 16
				# return CNN_3_GTSRB(input_shape=input_shape, mid_dim=mid_dim, num_classes=self.num_classes)
				else:
					mid_dim = 4
				return CNN_3_proto(input_shape=input_shape, mid_dim=mid_dim, num_classes=self.num_classes)

			if model is not None:
				model.to(self.device)
				return model
			else:
				raise Exception("Wrong model name")
		except Exception as e:
			print("create model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def get_parameters(self, config):
		parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
		return parameters

		# It does the same of "get_parameters", but using "get_parameters" in outside of the core of Flower is causing errors
	def get_parameters_of_model(self):
		parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
		return parameters

	def initial(self, parameters):
		parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
		for new_param, old_param in zip(parameters, self.model.parameters()):
			old_param.data = new_param.data.clone()

	def clone_model(self, model, target):
		for param, target_param in zip(model.parameters(), target.parameters()):
			target_param.data = param.data.clone()
		# target_param.grad = param.grad.clone()

	# def save_parameters(self):
	# 	filename = """./fedper_saved_weights/{}/{}/{}.json""".format(self.model_name, self.cid, self.cid)
	# 	weights = self.model.get_weights()
	# 	personalized_layers_weights = []
	# 	for i in range(self.n_personalized_layers):
	# 		personalized_layers_weights.append(weights[len(weights)-self.n_personalized_layers+i])
	# 	data = json.dumps([i.tolist() for i in personalized_layers_weights])
	# 	jsonFile = open(filename, "w")
	# 	jsonFile.write(data)
	# 	jsonFile.close()


	# def set_parameters_to_model(self, parameters):
	# 	parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
	# 	for new_param, old_param in zip(parameters, self.model.parameters()):
	# 		old_param.data = new_param.data.clone()

	def fit(self, parameters, config):
		try:
			selected_clients   = []
			trained_parameters = []
			selected           = 0
			mse_list = []
			train_loss = 0
			train_acc = 0
			protos = defaultdict(list)
			server_round = int(config['round'])

			start_time = time.process_time()
			if config['selected_clients'] != '':
				selected_clients = [int (cid_selected) for cid_selected in config['selected_clients'].split(' ')]
			if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:

				# the parameters are saved in a file because in each round new instances of client are created
				# if int(config['round']) == 1:
				# 	self.inicial(parameters)
				if self.cid in selected_clients or self.client_selection == False or server_round == 1:
					if self.dynamic_data != "no":
						self.trainloader, self.testloader, self.traindataset, self.testdataset = self.load_data(
							self.dataset,
							n_clients=self.n_clients, server_round=server_round)
				if int(config['round']) > 1:
					self.set_parameters_to_model()
					self.set_proto(parameters)

				selected = 1
				self.model.train()

				max_local_steps = self.local_epochs
				repeated_protos = 0

				for step in range(max_local_steps):
					train_num = 0
					train_acc = 0
					train_loss = 0
					macro_f1_score = 0
					weigthed_f1_score = 0
					micro_f1_score = 0
					for i, (x, y) in enumerate(self.trainloader):
						if type(x) == type([]):
							x[0] = x[0].to(self.device)
						else:
							x = x.to(self.device)

						y = np.array(y).astype(int)

						train_num += y.shape[0]

						self.optimizer.zero_grad()
						# rep = self.model.base(x)
						# output = self.model.head(rep)
						output, rep = self.model(x)
						y = torch.tensor(y)
						y = y.to(self.device)
						loss = self.loss(output, y)

						if self.global_protos != None:
							proto_new = np.zeros(rep.shape)
							for i, yy in enumerate(y):
								y_c = yy.item()
								if not np.isnan(self.global_protos[y_c]).any() or np.sum(self.global_protos[y_c] == 0):
									proto_new[i] = self.global_protos[y_c]
								else:
									proto_new[i] = rep[i, :].detach().data
									repeated_protos += 1

							proto_new = torch.Tensor(proto_new.tolist())
							mse_loss = self.loss_mse(proto_new, rep) * self.lamda
							# print("antes: ", loss)
							# print("aqui: ", mse_loss)
							loss += mse_loss
							mse_list.append(mse_loss.item())

						loss.backward()
						self.optimizer.step()

						for i, yy in enumerate(y):
							y_c = yy.item()
							protos[y_c].append(rep[i, :].detach().data)
							self.protos_samples_per_class[y_c] += 1

						train_loss += loss.item()
						train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
						macro_f1_score += f1_score(y, output.detach().numpy().tolist(), average='macro',
												   zero_division=1)
						weigthed_f1_score += f1_score(y, output.detach().numpy().tolist(), average='weighted',
													  zero_division=1)
						micro_f1_score += f1_score(y, output.detach().numpy().tolist(), average='micro',
												   zero_division=1)

				self.save_parameters()

				self.protos = self.agg_func(protos)

				for i in range(len(self.protos)):
					if np.sum(self.protos[i]) == 0 and self.protos_samples_per_class[i] != 0:
						print("errado")
						exit()

				total_time         = time.process_time() - start_time
				size_of_parameters = sum(map(sys.getsizeof, parameters))
				avg_loss_train     = train_loss/train_num
				avg_acc_train      = train_acc/train_num
				macro_f1_score = macro_f1_score / train_num
				weigthed_f1_score = weigthed_f1_score / train_num
				micro_f1_score = micro_f1_score / train_num

				data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train, macro_f1_score, weigthed_f1_score, micro_f1_score]

				self._write_output(
					filename=self.train_client_filename,
					data=data)

				# if self.global_protos is not None:
				# 	if np.isnan(self.global_protos[3]).any() or np.isnan(self.global_protos[4]).any():
				# 		print("proto nao nulo, rodada ", config['round'])
				# 		print("recebidos:")
				# 		print(self.global_protos[3])
				# 		exit()

				fit_response = {
					'cid': self.cid,
					'protos_samples_per_class': self.protos_samples_per_class,
					'proto': {i: np.array(self.protos[i]) for i in range(len(self.protos))}
				}

				# print("saindo: ", config['round'], " forma: ", self.protos[0].shape, self.protos[1].shape, self.protos[2].shape, self.protos[3].shape)
				return self.protos, train_num, fit_response
			else:
				# print("saiu", config['round'], self.cid)
				print("errou")
				return [np.zeros((100,))], 0, {
					'cid': self.cid,
					'protos_samples_per_class': {i: 0 for i in range(self.num_classes)},
					'proto': {i: np.zeros((1, 100)) for i in range(self.num_classes)}
				}
		except Exception as e:
			print("fit")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def evaluate(self, parameters, config):
		try:
			server_round = int(config['round'])
			self.set_proto(parameters)
			# loss, accuracy     = self.model.evaluate(self.x_test, self.y_test, verbose=0)
			self.set_parameters_to_model()
			if self.dynamic_data != "no":
				self.trainloader, self.testloader, self.traindataset, self.testdataset = self.load_data(
					self.dataset,
					n_clients=self.n_clients, server_round=server_round)
			self.model.eval()

			test_acc = 0
			test_loss = []
			test_mse_loss = []
			test_num = 0
			macro_f1_score = 0
			weigthed_f1_score = 0
			micro_f1_score = 0
			count = 0

			with torch.no_grad():
				for x, y in self.testloader:
					if type(x) == type([]):
						x[0] = x[0].to(self.device)
					else:
						x = x.to(self.device)
					self.optimizer.zero_grad()
					if type(y) == tuple:
						y = torch.from_numpy(np.array(y).astype(int))
					y = y.to(self.device)
					y = torch.tensor(y)
					output, rep = self.model(x)

					# prediciton based on similarity
					# output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
					#
					# for i, r in enumerate(rep):
					# 	for j in range(len(self.global_protos)):
					# 		pro = torch.Tensor(self.global_protos[j].tolist())
					# 		# print("global: ", pro.shape, " local: ", r.shape)
					# 		# print("saida mse: ", self.loss_mse(r, pro).shape)
					# 		# print("entrada: ", output[i, j].shape)
					# 		output[i, j] = self.loss_mse(r, pro)
					#
					# test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
					# test_loss.append(torch.sum(torch.min(output, dim=1)[0]).item())


					loss = self.loss(output, y)
					mse_value = 0
					if self.global_protos != None:
						proto_new = np.zeros(rep.shape)
						for i, yy in enumerate(y):
							y_c = yy.item()
							if not np.isnan(self.global_protos[y_c]).any() or not np.sum(self.global_protos[y_c] == 0):
								proto_new[i] = self.global_protos[y_c]
							else:
								proto_new[i] = rep[i, :].detach().data
						proto_new = torch.Tensor(proto_new.tolist())
						mse_loss = self.loss_mse(proto_new, rep) * self.lamda
						mse_value = mse_loss.item()
						test_mse_loss.append(mse_value)
					test_loss.append(loss.item() + mse_value)
					test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

					test_num += y.shape[0]
					count += 1
					macro_f1_score += f1_score(y, output.detach().numpy().tolist(), average='macro', zero_division=1)
					weigthed_f1_score += f1_score(y, output.detach().numpy().tolist(), average='weighted',
												  zero_division=1)
					micro_f1_score += f1_score(y, output.detach().numpy().tolist(), average='micro', zero_division=1)

			size_of_parameters = sum(map(sys.getsizeof, parameters))
			# print("test loss: ", test_loss)
			loss = np.mean(test_loss)
			macro_f1_score = macro_f1_score / count
			weigthed_f1_score = weigthed_f1_score / count
			micro_f1_score = micro_f1_score / count
			accuracy = test_acc/test_num
			size_of_config = sys.getsizeof(config)
			data = [config['round'], self.cid, size_of_parameters, size_of_config, loss, accuracy, macro_f1_score, weigthed_f1_score, micro_f1_score]

			self._write_output(filename=self.evaluate_client_filename,
							   data=data)

			evaluation_response = {
				"cid"      : self.cid,
				"accuracy" : float(accuracy),
				"macro f1-score": macro_f1_score,
				"weithed f1-score": weigthed_f1_score,
				"micro f1-score": micro_f1_score,
				"mse_loss" : np.mean(test_loss)
			}

			return loss, test_num, evaluation_response
		except Exception as e:
			print("evaluate")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model(self, parameters=[]):
		filename = """./fedproto_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid, self.cid)
		if os.path.exists(filename):
			self.model.load_state_dict(torch.load(filename))
		else:
			print("Model does not exist")
			pass

	def save_parameters(self):
		# os.makedirs("""{}/fedproto_saved_weights/{}/{}/""".format(os.getcwd(), self.model_name, self.cid),
		# 			exist_ok=True)
		try:
			filename = """./fedproto_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid)
			if Path(filename).exists():
				os.remove(filename)
			torch.save(self.model.state_dict(), filename)
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_proto(self, protos):
		self.global_protos = copy.deepcopy(protos)

	def agg_func(self, protos):
		"""
        Returns the average of the weights.
        """
		try:
			proto_shape = None
			for label in protos:
				proto_list = protos[label]

				if len(proto_list) > 1:
					proto = proto_list[0].detach().numpy()
					for i in range(1, len(proto_list)):
						proto += proto_list[i].detach().numpy()
					protos[label] = proto / len(proto_list)
					proto_shape = proto.shape
				else:
					protos[label] = proto_list[0].detach().numpy()

			if self.global_protos is not None:
				numpy_protos = copy.deepcopy(self.global_protos)
			else:
				numpy_protos = [np.zeros(proto_shape) for i in range(self.num_classes)]
			for label in protos:
				numpy_protos[label] = protos[label]

			return numpy_protos
		except Exception as e:
			print("agg fun")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)