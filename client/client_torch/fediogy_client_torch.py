from client.client_torch.client_base_torch import ClientBaseTorch


import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class FedYogiClientTorch(ClientBaseTorch):

	def __init__(self,
                 cid,
                 n_clients,
                 n_classes,
                 epochs=1,
                 model_name         = 'DNN',
                 client_selection   = False,
                 strategy_name      ='FedYogi',
                 aggregation_method = 'None',
                 dataset            = '',
                 perc_of_clients    = 0,
                 decay              = 0,
                 non_iid            = False,
                 new_clients		= False,
                 new_clients_train	= False):

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
                         non_iid=non_iid,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train)