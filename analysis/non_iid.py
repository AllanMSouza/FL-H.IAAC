from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from optparse import OptionParser
from base_plots import bar_plot, line_plot

class ComplexNumber:
    def __init__(self, num_clients, aggregation_method, model_name, strategy_name_list, dataset_name):
        self.n_clients = num_clients
        self.agg_method = aggregation_method
        self.model_name = model_name
        self.strategy_name_list = strategy_name_list
        self.dataset_name = dataset_name
        self.base_files_names = {'evaluate_client': 'evaluate_client.csv',
                                 'server': 'server.csv',
                                 'train_client': 'train_client.csv'}

        self.df_files_names = {'evaluate_client': None,
                               'server': None,
                               'train_client': None}

    def start(self):

        models_directories = {self.strategy_name_list[i]: """{}/{}-{}/{}/{}/{}/""".format('/home/claudio/Documentos/pycharm_projects/FedLTA/logs',
                                                                                            self.strategy_name_list[i],
                                                                                           self.agg_method,
                                                                                           self.n_clients,
                                                                                           self.model_name,
                                                                                           self.dataset_name) for i in range(len(self.strategy_name_list))}

        # read datasets
        print(models_directories)
        for i in range(len(self.strategy_name_list)):
            strategy_name = self.strategy_name_list[i]

            for j in self.base_files_names:
                file_name = self.base_files_names[j]
                df = pd.read_csv(models_directories[strategy_name]+file_name)
                df['Strategy name'] = np.array([strategy_name]*len(df))
                if i == 0:
                    self.df_files_names[j] = df
                else:
                    self.df_files_names[j] = pd.concat([self.df_files_names[j], df], ignore_index=True)

        self.server_analysis()
        self.evaluate_client_analysis()


    def server_analysis(self):

        # server analysis
        df = self.df_files_names['server']
        base_dir = '/home/claudio/Documentos/pycharm_projects/FedLTA/analysis/output/'
        x_column = 'Server round'
        y_column = 'Accuracy aggregated'
        hue = 'Strategy name'
        title = ""
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="server_acc_round_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue)

        x_column = 'Server round'
        y_column = 'Time'
        hue = 'Strategy name'
        title = ""
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="server_time_round_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue)

    def evaluate_client_analysis(self):
        # acc
        df = self.df_files_names['evaluate_client']
        base_dir = '/home/claudio/Documentos/pycharm_projects/FedLTA/analysis/output/'
        x_column = 'Round'
        y_column = 'Accuracy'
        hue = 'Strategy name'
        title = ""
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_acc_round_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue)

        # loss
        x_column = 'Round'
        y_column = 'Loss'
        hue = 'Strategy name'
        title = ""
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_loss_round_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue)

        # size of parameters
        print(df)
        def strategy(df):
            parameters = int(df['Size of parameters'].mean())

            return pd.DataFrame({'Size of parameters': [parameters]})
        df_test = df[['Round', 'Size of parameters', 'Strategy name']].groupby('Strategy name').apply(lambda e: strategy(e)).reset_index()[['Size of parameters', 'Strategy name']]
        df_test.to_csv(base_dir+"evaluate_client_size_of_parameters_round.csv", index=False)
        print(df_test)
        x_column = 'Strategy name'
        y_column = 'Size of parameters'
        hue = None
        title = ""
        bar_plot(df=df_test,
                  base_dir=base_dir,
                  file_name="evaluate_client_size_of_parameters_round_barplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue)





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
    parser.add_option("-s", "--strategy", dest="strategy_name", default='FedSGD',
                      help="Strategy of the federated learning", metavar="STR")
    parser.add_option("-a", "--aggregation_method", dest="aggregation_method", default='None',
                      help="Algorithm used for selecting clients", metavar="STR")
    parser.add_option("-m", "--model", dest="model_name", default='DNN', help="Model used for trainning", metavar="STR")
    parser.add_option("-d", "--dataset", dest="dataset", default='MNIST', help="Dataset used for trainning",
                      metavar="STR")
    parser.add_option("-r", "--round", dest="rounds", default=5, help="Number of communication rounds", metavar="INT")

    (opt, args) = parser.parse_args()

    strategy_name_list = ['FedAVG', 'FedProto']

    # noniid = NonIID(int(opt.n_clients), opt.aggregation_method, opt.model_name, strategy_name_list, opt.dataset)
    # noniid.start()
    c = ComplexNumber(int(opt.n_clients), opt.aggregation_method, opt.model_name, strategy_name_list, opt.dataset)
    print(c.n_clients, " ", c.strategy_name_list)
    c.start()