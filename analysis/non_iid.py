from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from optparse import OptionParser
from base_plots import bar_plot, line_plot
from pathlib import Path
import os
import ast

class NonIid:
    def __init__(self, num_clients, aggregation_method, perc_of_clients, non_iid, model_name, strategy_name_list, dataset_name, new_clients,new_clients_train, experiment, comment, epochs):
        self.n_clients = num_clients
        self.aggregation_method = aggregation_method
        self.perc_of_clients = perc_of_clients
        self.non_iid = non_iid
        self.model_name = model_name
        self.strategy_name_list = strategy_name_list
        self.dataset_name = dataset_name
        self.new_clients = new_clients
        self.new_clients_train = new_clients_train
        self.experiment = experiment
        self.comment = comment
        self.epochs = epochs
        self.base_files_names = {'evaluate_client': 'evaluate_client.csv',
                                 'server': 'server.csv',
                                 'train_client': 'train_client.csv'}

        self.df_files_names = {'evaluate_client': None,
                               'server': None,
                               'train_client': None}

    def _get_strategy_config(self, strategy_name):
        if self.aggregation_method == 'POC':
            strategy_config = f"{strategy_name}-{self.aggregation_method}-{self.perc_of_clients}"

        # elif self.aggregation_method == 'FedLTA':
        #     strategy_config = f"{self.strategy_name}-{self.aggregation_method}-{self.decay_factor}"

        elif self.aggregation_method == 'None':
            strategy_config = f"{strategy_name}-{self.aggregation_method}"

        print("antes: ", self.aggregation_method, strategy_config)

        return strategy_config

    def start(self):

        self.base_dir = """analysis/output/experiment_{}/{}/new_clients_{}_train_{}/{}_clients/{}/{}_local_epochs/{}/""".format(self.experiment,
                                                                                            self.aggregation_method+str(self.perc_of_clients),
                                                                                            self.new_clients,
                                                                                              self.new_clients_train,
                                                                                            self.n_clients,
                                                                                            self.dataset_name,
                                                                                            self.epochs,
                                                                                            self.comment)

        if not Path(self.base_dir).exists():
            os.makedirs(self.base_dir+"csv")
            os.makedirs(self.base_dir + "png/")
            os.makedirs(self.base_dir + "svg/")

        models_directories = {self.strategy_name_list[i]:
                              """{}/{}/new_clients_{}_train_{}/{}/{}/{}/{}_local_epochs/""".
                              format('/home/claudio/Documentos/pycharm_projects/FedLTA/logs',
                                     self._get_strategy_config(self.strategy_name_list[i]),
                                     self.new_clients,
                                     self.new_clients_train,
                                     self.n_clients,
                                     self.model_name,
                                     self.dataset_name,
                                     self.epochs) for i in range(len(self.strategy_name_list))}

        # read datasets
        print(models_directories)
        for i in range(len(self.strategy_name_list)):
            strategy_name = self.strategy_name_list[i]

            for j in self.base_files_names:
                file_name = self.base_files_names[j]
                df = pd.read_csv(models_directories[strategy_name]+file_name)
                df['Strategy'] = np.array([strategy_name]*len(df))
                if i == 0:
                    self.df_files_names[j] = df
                else:
                    self.df_files_names[j] = pd.concat([self.df_files_names[j], df], ignore_index=True)

        self.server_analysis()
        self.evaluate_client_analysis()


    def server_analysis(self):

        # server analysis
        df = self.df_files_names['server']
        df['Accuracy aggregated (%)'] = df['Accuracy aggregated'] * 100
        df['Time (seconds)'] = df['Time'].to_numpy()
        x_column = 'Server round'
        y_column = 'Accuracy aggregated (%)'
        hue = 'Strategy'
        title = ""
        line_plot(df=df,
                  base_dir=self.base_dir,
                  file_name="server_acc_round_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue)

        x_column = 'Server round'
        y_column = 'Time (seconds)'
        hue = 'Strategy'
        title = ""
        line_plot(df=df,
                  base_dir=self.base_dir,
                  file_name="server_time_round_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue)

    def evaluate_client_analysis(self):
        # acc
        df = self.df_files_names['evaluate_client']
        df['Accuracy (%)'] = df['Accuracy'] * 100
        x_column = 'Round'
        y_column = 'Accuracy (%)'
        hue = 'Strategy'
        title = ""
        line_plot(df=df,
                  base_dir=self.base_dir,
                  file_name="evaluate_client_acc_round_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue)

        # loss
        x_column = 'Round'
        y_column = 'Loss'
        hue = 'Strategy'
        title = ""
        line_plot(df=df,
                  base_dir=self.base_dir,
                  file_name="evaluate_client_loss_round_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue)

        # size of parameters
        print(df)
        def strategy(df):
            parameters = int(df['Size of parameters'].mean())

            return pd.DataFrame({'Size of parameters (bytes)': [parameters]})
        df_test = df[['Round', 'Size of parameters', 'Strategy']].groupby('Strategy').apply(lambda e: strategy(e)).reset_index()[['Size of parameters (bytes)', 'Strategy']]
        df_test.to_csv(self.base_dir+"csv/evaluate_client_size_of_parameters_round.csv", index=False)
        print(df_test)
        x_column = 'Strategy'
        y_column = 'Size of parameters (bytes)'
        hue = None
        title = "Two layers"
        bar_plot(df=df_test,
                  base_dir=self.base_dir,
                  file_name="evaluate_client_size_of_parameters_round_barplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                 sci=True)
        bar_plot(df=df_test,
                 base_dir=self.base_dir,
                 file_name="evaluate_client_size_of_parameters_round_barplot",
                 x_column=x_column,
                 y_column=y_column,
                 title=title,
                 hue=hue,
                 log_scale=True,
                 sci=True)





# class NonIID:
#     def __int__(self, num_clients, aggregation_method, model_name, strategy_name_list, dataset_name):
#         self.num_clients = num_clients
#         self.aggregation_method = aggregation_method
#         self.model_name = model_name
#         self.strategy_name = strategy_name_list
#         self.dataset = dataset_name
#         self.base_files_names = {'evaluate_client': 'evaluate_client.csv',
#                                  'server': 'server.csv',
#                                  'train_client': 'train_client.csv'}
#
#         self.df_files_names = {'evaluate_client': None,
#                                  'server': None,
#                                  'train_client': None}
#
#
#     def start(self):
#
#         models_directories = {self.model_name[i]: """{}-{}/{}/{}/{}/""".format(self.strategy_name,
#                                                                                self.aggregation_method,
#                                                                                self.num_clients,
#                                                                                self.model_name[i],
#                                                                                self.dataset) for i in range(len(self.model_name))}
#
#         # read datasets
#         for i in range(len(self.model_name)):
#             model_name = self.model_name[i]
#
#             for j in self.base_files_names:
#                 file_name = self.base_files_names[j]
#                 df = pd.read_csv(models_directories[i]+file_name)
#                 df['Strategy name'] = np.array([model_name]*len(df))
#                 if i == 0:
#                     self.df_files_names[j] = df
#                 else:
#                     self.df_files_names[j] = pd.concat(self.df_files_names[j], df, ignore_index=True)
#
#         self.server_analysis()
#
#
#     def server_analysis(self):
#
#         df = self.df_files_names['server']
#         print(df)


if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("-c", "--clients", dest="n_clients", default=10, help="Number of clients in the simulation",
                      metavar="INT")
    parser.add_option("-a", "--aggregation_method", dest="aggregation_method", default='None',
                      help="Algorithm used for selecting clients", metavar="STR")
    parser.add_option("-m", "--model", dest="model_name", default='DNN', help="Model used for trainning", metavar="STR")
    parser.add_option("-d", "--dataset", dest="dataset", default='MNIST', help="Dataset used for trainning",
                      metavar="STR")
    parser.add_option("", "--non-iid", dest="non_iid", default=False,
                      help="whether or not it was non iid experiment")
    parser.add_option("", "--poc", dest="poc", default=1.,
                      help="percentage of clients to fit", metavar="FLOAT")
    parser.add_option("-r", "--round", dest="rounds", default=5, help="Number of communication rounds", metavar="INT")
    parser.add_option("", "--new_clients", dest="new_clients", default='False', help="Adds new clients after a specific round")
    parser.add_option("", "--new_clients_train", dest="new_clients_train", default='False',
                      help="")
    parser.add_option("--strategy", action='append', dest="strategies", default=[])
    parser.add_option("--experiment",  dest="experiment", default='')
    parser.add_option("--comment", dest="comment", default='')
    parser.add_option("--epochs", dest="epochs", default=1)

    (opt, args) = parser.parse_args()

    # strategy_name_list = ['FedAVG', 'FedAvgM', 'FedClassAvg''QFedAvg', 'FedPer', 'FedProto', 'FedYogi', 'FedLocal']
    strategy_name_list = opt.strategies

    # noniid = NonIID(int(opt.n_clients), opt.aggregation_method, opt.model_name, strategy_name_list, opt.dataset)
    # noniid.start()
    c = NonIid(int(opt.n_clients), opt.aggregation_method, float(opt.poc), ast.literal_eval(opt.non_iid), opt.model_name, strategy_name_list, opt.dataset, ast.literal_eval(opt.new_clients), ast.literal_eval(opt.new_clients_train), opt.experiment, opt.comment, opt.epochs)
    print(c.n_clients, " ", c.strategy_name_list)
    c.start()