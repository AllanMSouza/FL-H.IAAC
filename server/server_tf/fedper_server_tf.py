from server.common_base_server import FedAvgBaseServer
from pathlib import Path
import shutil

class FedPerServerTf(FedAvgBaseServer):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 num_epochs,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
                 strategy_name='FedPer',
                 model_name='',
                 new_clients=False,
                 new_clients_train=False):

        super().__init__(aggregation_method=aggregation_method,
                         n_classes=n_classes,
                         fraction_fit=fraction_fit,
                         num_clients=num_clients,
                         num_rounds=num_rounds,
                         num_epochs=num_epochs,
                         decay=decay,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name=strategy_name,
                         model_name=model_name,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train,
                         type='tf')

        self.create_folder()


    def create_folder(self):

        directory = """fedper_saved_weights/{}/""".format(self.model_name)
        if Path(directory).exists():
            shutil.rmtree(directory)
        for i in range(self.num_clients):
            Path("""fedper_saved_weights/{}/{}/""".format(self.model_name, i)).mkdir(parents=True, exist_ok=True)
