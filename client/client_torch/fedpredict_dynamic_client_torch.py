from client.client_torch import FedAvgClientTorch
from ..fedpredict_core import fedpredict_dynamic_client
from torch.nn.parameter import Parameter
import torch
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
import ast
from utils.compression_methods.sparsification import calculate_bytes, sparse_bytes, sparse_matrix

import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class FedPredictDynamicClientTorch(FedAvgClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedPredict_Dynamic',
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

		self.m_combining_layers = [i for i in range(len([i for i in self.create_model().parameters()]))]
		self.similarity = 1
		self.global_model = self.create_model().to(self.device)
		self.lr_loss = torch.nn.MSELoss()
		self.clone_model = self.create_model().to(self.device)
		self.round_of_last_fit = 0
		self.rounds_of_fit = 0
		self.T = int(args.T)
		self.accuracy_of_last_round_of_fit = 0
		self.start_server = 0
		self.client_information_filename = """{}_saved_weights/{}/{}/{}.csv""".format(strategy_name.lower(), self.model_name, self.cid, self.cid)
		self.client_information_file = self.read_client_file()
		self.filename = """./{}_saved_weights/{}/{}/model.pth""".format(strategy_name.lower(), self.model_name, self.cid)
		self.global_model_filename = """./{}_saved_weights/{}/{}/global_model.pth""".format(strategy_name.lower(), self.model_name,
																		self.cid)


	def save_client_information_fit(self, server_round, acc_of_last_fit, predictions):

		try:
			self.classes_proportion, self.imbalance_level = self._calculate_classes_proportion()
			df = pd.read_csv(self.client_information_filename)
			row = df.iloc[0]
			if int(row['first_round']) == -1:
				first_round = -1
			else:
				first_round = int(row['first_round'])

			pd.DataFrame(
				{'current_round': [server_round], 'classes_distribution': [str(self.classes_proportion)], 'round_of_last_fit': [server_round],
				 'round_of_last_evaluate': [-1], 'acc_of_last_fit': [acc_of_last_fit], 'first_round': [first_round],
				 'acc_of_last_evaluate': [0]}).to_csv(self.client_information_filename,
													  index=False)

		except Exception as e:
			print("save client information fit")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


	def save_parameters(self):
		# Using 'torch.save'
		try:
			if Path(self.filename).exists():
				os.remove(self.filename)
			torch.save(self.model.state_dict(), self.filename)
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_parameters_global_model(self, global_model):
		# Using 'torch.save'
		try:
			# filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid)
			if Path(self.global_model_filename).exists():
				os.remove(self.global_model_filename)

			parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_model]
			for new_param, old_param in zip(parameters, self.global_model.parameters()):
				old_param.data = new_param.data.clone()
			torch.save(self.global_model.state_dict(), self.filename)
		except Exception as e:
			print("save parameters global model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	# def _calculate_classes_proportion(self):
	#
	# 	try:
	# 		traindataset = self.traindataset
	# 		y_train = list(traindataset.targets)
	# 		proportion = np.array([0] * self.num_classes)
	#
	# 		for i in y_train:
	#
	# 			proportion[i] += 1
	#
	# 		proportion = proportion/np.sum(proportion)
	#
	# 		return proportion
	#
	# 	except Exception as e:
	# 		print("calculate classes proportion")
	# 		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	# def _detect_context_change(self):
	#
	# 	similarity_between_contexts, imbalance_level, fraction_of_classes = self._calculate_contexts_similarities()
	#
	# 	threshold = 0.8
	#
	# 	print("similaridade de contextos: ", similarity_between_contexts)
	#
	# 	return similarity_between_contexts, imbalance_level, fraction_of_classes

	def _calculate_contexts_similarities(self):

		try:
			"""
				It measures the cosine similarity between the last and current class distribution of the local dataset
			"""
			n = len(self.client_information_file['classes_distribution'])
			# print("antes ", self.client_information_file['classes_distribution'].tolist()[-1])
			last_proportion = ast.literal_eval(self.client_information_file['classes_distribution'].tolist()[-1])
			last_training = self.client_information_file['round_of_last_fit'].tolist()[-1]

			current_proportion, imbalance_level = self._calculate_classes_proportion()
			fraction_of_classes = sum([1 if i > 0 else 0 for i in current_proportion])/self.num_classes

			if len(last_proportion) != len(current_proportion) or last_training == -1:
				return 1, imbalance_level, fraction_of_classes, current_proportion

			last_proportion = np.array(last_proportion)
			current_proportion = np.array(current_proportion)

			if (last_proportion == current_proportion).all():
				print("igual")

				cosine_similarity = 1

			else:
				print("diferente ")
				dot_product = np.dot(last_proportion, current_proportion)

				norm_vector1 = np.linalg.norm(last_proportion)

				norm_vector2 = np.linalg.norm(current_proportion)

				cosine_similarity = dot_product / (norm_vector1 * norm_vector2)

			return cosine_similarity, imbalance_level, fraction_of_classes, current_proportion

		except Exception as e:
			print("calculate contexts similarities")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def read_client_file(self):

		try:

			df = pd.read_csv(self.client_information_filename)

			return df

		except Exception as e:
			print("read client file")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	# def fit(self, parameters, config):
	# 	try:
	# 		trained_parameters, train_num, fit_response = super().fit(parameters, config)
	# 		server_round = int(config['round'])
	#
	# 		if self.dynamic_data != "no":
	# 			self.trainloader, self.testloader, self.traindataset, self.testdataset = self.load_data(self.dataset,
	# 																								n_clients=self.n_clients, server_round=server_round)
	#
	#
	# 		fit_response = {
	# 			'cid': self.cid,
	# 			'local_classes': self.classes_proportion
	# 		}
	#
	# 		return trained_parameters, train_num, fit_response
	# 	except Exception as e:
	# 		print("fit fedpredict dynamic")
	# 		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model_evaluate(self, global_parameters, config={}):
		# Using 'torch.load'
		try:
			local_classes = self.classes_proportion

			server_round = config['round']
			similarity, imbalance_level, fraction_of_classes, current_proportion = self._calculate_contexts_similarities()
			self.current_proportion = np.array(current_proportion)
			self.similarity = similarity
			# print("similaridade: ", similarity, ' imbalance level: ', imbalance_level, ' fraction of classes:', fraction_of_classes)
			if config['round'] >= int(0.7*self.n_rounds):
				if similarity == 1:
					print("cliente ", self.cid, " usou o modelo local rodada ", config['round'])
				else:
					print("cliente ", self.cid, " usou o modelo global rodada ", config['round'])
			row = self.clients_pattern.query("""Round == {} and Cid == {}""".format(server_round, self.cid))[
				'Pattern'].tolist()
			if len(row) != 1:
				raise ValueError(
					"""Pattern not found for client {}. The pattern may not exist or is duplicated""".format(self.cid))
			pattern = int(row[0])
			if config['round'] >= int(0.7*self.n_rounds):
				pattern = config['pattern']
				print("""cliente {} mudou padrao {}""".format(self.cid, pattern))
			local_data_information = {'similarity': similarity, 'imbalance_level': imbalance_level, 'fraction_of_classes': fraction_of_classes}
			self.model = fedpredict_dynamic_client(self.filename, self.model, global_parameters, config, mode=None, local_client_information=local_data_information, current_proportion=current_proportion, pattern=pattern, cid=self.cid)

		except Exception as e:
			print("Set parameters to model evaluate dyn")
			print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, self.cid), type(e).__name__, e)

	def model_eval(self):
		try:
			self.model.to(self.device)
			self.model.eval()

			test_acc = 0
			test_loss = 0
			test_num = 0

			predictions = np.array([])
			labels = np.array([])

			with torch.no_grad():
				for x, y in self.testloader:
					if type(x) == type([]):
						x[0] = x[0].to(self.device)
					else:
						x = x.to(self.device)
					# if self.dataset == 'EMNIST':
					# 	x = x.view(-1, 28 * 28)
					if type(y) == tuple:
						y = torch.from_numpy(np.array(y).astype(int))
					y = torch.from_numpy(np.array(y).astype(int))
					self.optimizer.zero_grad()
					y = y.to(self.device)
					y = torch.tensor(y)
					output = self.model(x)

					print("""cliente {} rodada {} simi {}""".format(self.cid, self.server_round, self.similarity))
					if self.similarity != 1:
						output = torch.multiply(output, torch.from_numpy(self.current_proportion * (1 - self.similarity)))
					loss = self.loss(output, y)
					test_loss += loss.item() * y.shape[0]
					prediction = torch.argmax(output, dim=1)
					predictions = np.append(predictions, prediction.cpu())
					labels = np.append(labels, y.cpu())
					test_acc += (torch.sum(prediction == y)).item()
					test_num += y.shape[0]

			loss = test_loss / test_num
			accuracy = test_acc / test_num

			return loss, accuracy, test_num, predictions, output, labels
		except Exception as e:
			print("model_eval")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def calculate_bytes(self, parameters):

		try:

			size = 0

			if self.comment == "sparsification":
				for p in parameters:
					aux = p[p==0]
					# print("quantidade zeros: ", len(aux))
					sparse = sparse_matrix(p)
					# print("Tamanho original: ", p.nbytes)
					b = sparse_bytes(sparse)
					# print("Apos esparcificacao: ", b)
					b = min(p.nbytes, b)
					size += b
			else:
				for p in parameters:
					size += p.nbytes
			return size

		except Exception as e:
			print("calculate bytes")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

