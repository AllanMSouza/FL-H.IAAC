from client.client_torch import FedKDClientTorch
from ..fedpredict_core import fedpredict_client
from torch.nn.parameter import Parameter
import torch
from pathlib import Path
import numpy as np
from utils.compression_methods.parameters_svd import inverse_parameter_svd_reading, parameter_svd_write
import os
import sys
import time
import copy
from models.torch import Logistic, DNN_student, DNN_teacher, CNNDistillation
from torchvision import models
import torch.nn.functional as F

import warnings
warnings.simplefilter("ignore")


# logging.getLogger("torch").setLevel(logging.ERROR)

def if_reduces_size(shape, n_components, dtype=np.float64):

    try:
        size = np.array([1], dtype=dtype)
        p = shape[0]
        q = shape[1]
        k = n_components

        if p*k + k*k + k*q < p*q:
            return True
        else:
            return False

    except Exception as e:
        print("svd")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

def get_size(parameter):
	try:
		# #print("recebeu: ", parameter.shape, parameter.ndim)
		# if type(parameter) == np.float32:
		# 	#print("caso 1: ", map(sys.getsizeof, parameter))
		# 	return map(sys.getsizeof, parameter)
		# if parameter.ndim <= 2:
		# 	#print("Caso 2: ", sum(map(sys.getsizeof, parameter)))
		# 	return sum(map(sys.getsizeof, parameter))
		#
		# else:
		# 	tamanho = 0
		# 	#print("Caso 3")
		# 	for i in range(len(parameter)):
		# 		tamanho += get_size(parameter[i])
		#
		# 	return tamanho
		return parameter.nbytes
	except Exception as e:
		print("get_size")
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

class FedKDClientTorch(FedKDClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedKD_with_FedPredict',
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

	def set_parameters_to_model_evaluate(self, global_parameters, config={}):
		# Using 'torch.load'
		try:
			self.model = fedpredict_client(self.filename, self.model, global_parameters, config)
		except Exception as e:
			print("Set parameters to model")
			print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, self.cid), type(e).__name__, e)
