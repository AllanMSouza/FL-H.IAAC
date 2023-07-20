from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from optparse import OptionParser
from base_plots import bar_plot, line_plot, ecdf_plot
from pathlib import Path
import os
import ast

class NonIid:
    def __init__(self, num_clients, aggregation_method, perc_of_clients, fraction_fit, non_iid, model_name, strategy_name_list, dataset_name, new_clients, new_clients_train, experiment, comment, epochs, type, decay, args):
        self.n_clients = num_clients
        self.aggregation_method = aggregation_method
        self.perc_of_clients = perc_of_clients
        self.fraction_fit = fraction_fit
        self.non_iid = non_iid
        self.model_name = model_name
        self.strategy_name_list = strategy_name_list
        self.dataset_name = dataset_name
        self.new_clients = new_clients
        self.new_clients_train = new_clients_train
        self.experiment = experiment
        self.comment = comment
        self.rounds = args.rounds
        self.class_per_client = int(args.class_per_client)
        self.alpha = float(args.alpha)
        self.layer_selection_evaluate = int(args.layer_selection_evaluate)
        self.epochs = epochs
        self.decay = decay
        self.type = type
        self.base_files_names = {'evaluate_client': 'evaluate_client.csv',
                                 'server': 'server.csv',
                                 'train_client': 'train_client.csv',
                                 'server_nt_acc': 'server_nt_acc.csv',
                                 'similarity': 'similarity_between_layers.csv'}

        self.df_files_names = {'evaluate_client': None,
                               'server': None,
                               'train_client': None,
                               'server_nt_acc': None,
                               'similarity': None}

    def _get_strategy_config(self, strategy_name):
        if self.aggregation_method == 'POC':
            strategy_config = f"{strategy_name}-{self.aggregation_method}-{self.perc_of_clients}"

        elif self.aggregation_method == 'FedLTA':
            strategy_config = f"{strategy_name}-{self.aggregation_method}-{self.decay}"

        elif self.aggregation_method == 'None':
            strategy_config = f"{strategy_name}-{self.aggregation_method}-{self.fraction_fit}"

        print("antes: ", self.aggregation_method, strategy_config)

        return strategy_config

    def start(self, title):
        self.base_dir = """analysis/output/{}/experiment_{}/{}/new_clients_{}_train_{}/{}_clients/{}/{}/classes_per_client_{}/alpha_{}/{}_rounds/{}_local_epochs/{}_comment/""".format(self.type,
                                                                                                                                    self.experiment,
                                                                                                                                    self.aggregation_method+str(self.fraction_fit),
                                                                                                                                    self.new_clients,
                                                                                                                                    self.new_clients_train,
                                                                                                                                    self.n_clients,
                                                                                                                                    self.dataset_name,
                                                                                                                                    self.model_name,
                                                                                                                                    self.class_per_client,
                                                                                                                                    self.alpha,
                                                                                                                                    self.rounds,
                                                                                                                                    self.epochs,
                                                                                                                                    self.comment)

        if not Path(self.base_dir).exists():
            os.makedirs(self.base_dir+"csv")
            os.makedirs(self.base_dir + "png/")
            os.makedirs(self.base_dir + "svg/")

        models_directories = {self.strategy_name_list[i]:
                              """{}/{}/{}/new_clients_{}_train_{}/{}/{}/{}/classes_per_client_{}/alpha_{}/{}_rounds/{}_local_epochs/{}_comment/{}_layer_selection_evaluate/""".
                              format(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/FL-H.IAAC/logs",
                                     self.type,
                                     self._get_strategy_config(self.strategy_name_list[i]),
                                     self.new_clients,
                                     self.new_clients_train,
                                     self.n_clients,
                                     self.model_name,
                                     self.dataset_name,
                                     self.class_per_client,
                                     self.alpha,
                                     self.rounds,
                                     self.epochs,
                                     self.comment,
                                     self.layer_selection_evaluate) for i in range(len(self.strategy_name_list))}

        # read datasets
        print(models_directories)
        for i in range(len(self.strategy_name_list)):
            strategy_name = self.strategy_name_list[i]

            for j in self.base_files_names:
                file_name = self.base_files_names[j]
                if 'similarity' in file_name and strategy_name != 'FedPredict':
                    continue
                df = pd.read_csv(models_directories[strategy_name]+file_name)
                df['Strategy'] = np.array([strategy_name]*len(df))
                if i == 0:
                    self.df_files_names[j] = df
                else:
                    self.df_files_names[j] = pd.concat([self.df_files_names[j], df], ignore_index=True)

        print("teste: ", self.df_files_names['evaluate_client'])
        self.server_nt_acc_analysis()
        self.server_analysis(title)
        self.evaluate_client_analysis()
        print("chamar")
        if "FedPredict" in self.strategy_name_list:
            print("entrou")
            self.similarity_analysis("Alpha=" + str(self.alpha))

    def server_nt_acc_analysis(self):

        strategies = self.df_files_names['server_nt_acc']['Strategy'].unique().tolist()

        for strategy in strategies:
            self.server_nt_acc_analyse(strategy)

    def server_nt_acc_analyse(self, strategy):

        df = self.df_files_names['server_nt_acc']
        x_column = 'Round'
        y_column = 'Accuracy (%)'
        hue = 'nt'

        nt_values = [0, 2, 4]
        hue_order = nt_values
        df = df.query("Strategy == '" + strategy + "' and nt in " + str(nt_values))
        title = strategy
        filename = "server_nt_acc_" + str(int(self.experiment) - 1) + "_" + strategy
        line_plot(df=df,
                  base_dir=self.base_dir,
                  file_name=filename,
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  type='1',
                  hue=hue,
                  hue_order=hue_order)

    def server_analysis(self, title):

        # server analysis
        df = self.df_files_names['server']
        df['Accuracy aggregated (%)'] = df['Accuracy aggregated'] * 100
        df['Time (seconds)'] = df['Time'].to_numpy()
        x_column = 'Server round'
        y_column = 'Accuracy aggregated (%)'
        hue = 'Strategy'
        line_plot(df=df,
                  base_dir=self.base_dir,
                  file_name="server_acc_round_lineplot_" + str(int(self.experiment) - 1),
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  y_lim=True,
                  y_max=100)

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
        df['Accuracy (%)'] = df['Accuracy (%)'].round(4)
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
                  hue=hue,
                  y_lim=True,
                  y_min=0,
                  y_max=100)

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

        ecdf_plot(df=df, base_dir=self.base_dir,
                  file_name="""evaluate_client_ecdf_accuracy""".format(),
                  x_column='Accuracy (%)', y_column=None, title='CDF',
                  y_lim=True, y_max=1)

        # size of parameters
        print(df)
        def strategy(df):
            parameters = float(df['Size of parameters'].mean())/1000000
            config = float(df['Size of config'].mean())/1000000
            total_size = parameters + config

            return pd.DataFrame({'Size of parameters (MB)': [parameters], 'Communication cost (MB)': [total_size]})
        df_test = df[['Round', 'Size of parameters', 'Size of config', 'Strategy']].groupby('Strategy').apply(lambda e: strategy(e)).reset_index()[['Size of parameters (MB)', 'Communication cost (MB)', 'Strategy']]
        df_test.to_csv(self.base_dir+"csv/evaluate_client_size_of_parameters_round.csv", index=False)
        strategies = df_test['Strategy'].unique().tolist()
        hue_order = []
        reference = []
        for i in range(len(strategies)):
            if "FedPredict" in strategies[i]:
                reference = [strategies[i]]
            else:
                hue_order.append(strategies[i])
        hue_order = reference + hue_order
        print("ordem: ", hue_order)
        df_test = df_test.sort_values('Size of parameters (MB)')
        print(df_test)
        print(self.base_dir)
        x_column = 'Strategy'
        y_column = 'Size of parameters (MB)'
        hue = None
        title = "Size of parameters (MB)"
        bar_plot(df=df_test,
                  base_dir=self.base_dir,
                  file_name="evaluate_client_size_of_parameters_round_barplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                 hue_order=hue_order,
                 y_lim=True,
                 sci=True)

        y_column = 'Communication cost (MB)'
        hue = None
        title = "Communication cost (MB)"
        bar_plot(df=df_test,
                 base_dir=self.base_dir,
                 file_name="evaluate_client_communication_cost_round_barplot",
                 x_column=x_column,
                 y_column=y_column,
                 title=title,
                 hue=hue,
                 hue_order=hue_order,
                 y_lim=True,
                 sci=True)

    def similarity_analysis(self, title):

        df = self.df_files_names['similarity'].drop_duplicates()
        df = df.query("Similarity > 0")
        df['Similarity'] = df['Similarity'].round(4)
        df['Layer'] = df['Layer'].astype(int) + 1
        print("ddd: \n", df)
        x_column = 'Server round'
        y_column = 'Similarity'
        hue = 'Layer'
        hue_order = np.sort(df[hue].unique().tolist()).tolist()
        print("ordem: ", hue_order)
        type = 1
        y_min = {'EMNIST': 0.9, 'CIFAR-10': 0.9}[dataset]
        line_plot(df=df,
                  base_dir=self.base_dir,
                  file_name="similarity_between_layers_per_round_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  hue_order=hue_order,
                  type=type,
                  y_lim=True,
                  y_min=y_min)

        x_column = 'Server round'
        y_column = 'Similarity'
        type = 1
        line_plot(df=df,
                  base_dir=self.base_dir,
                  file_name="mean_similarity_between_layers_per_round_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  y_lim=True,
                  y_min=y_min)






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
    parser.add_option("", "--type", dest="type", default='torch',
                      help="")
    parser.add_option("--strategy", action='append', dest="strategies", default=[])
    parser.add_option("--experiment",  dest="experiment", default='')
    parser.add_option("--decay", dest="decay", default=0)
    parser.add_option("--comment", dest="comment", default='')
    parser.add_option("--epochs", dest="epochs", default=1)
    parser.add_option("", "--fraction_fit", dest="fraction_fit", default=0,
                      help="fraction of selected clients to be trained", metavar="FLOAT")
    parser.add_option("--class_per_client", help="Number of classes per client", default=2)
    parser.add_option("--alpha", help="Dirichlet alpha parameter", default=0.1)
    parser.add_option("--layer_selection_evaluate", help="", default=0)

    (opt, args) = parser.parse_args()

    # strategy_name_list = ['FedAVG', 'FedAvgM', 'FedClassAvg''QFedAvg', 'FedPer', 'FedProto', 'FedYogi', 'FedLocal']
    strategy_name_list = opt.strategies

    # noniid = NonIID(int(opt.n_clients), opt.aggregation_method, opt.model_name, strategy_name_list, opt.dataset)
    # noniid.start()
    c = NonIid(num_clients=int(opt.n_clients), aggregation_method=opt.aggregation_method, perc_of_clients=float(opt.poc), fraction_fit=float(opt.fraction_fit), non_iid=ast.literal_eval(opt.non_iid),
               model_name=opt.model_name, strategy_name_list=strategy_name_list, dataset_name=opt.dataset, new_clients=ast.literal_eval(opt.new_clients),
               new_clients_train=ast.literal_eval(opt.new_clients_train), experiment=opt.experiment, comment=opt.comment, epochs=opt.epochs, type=opt.type, decay=opt.decay, args=opt)

    print(c.n_clients, " ", c.strategy_name_list)
    if opt.comment == '':
        comment = 'bottom up'
    else:
        comment = 'top down'
    dataset = opt.dataset
    if dataset == 'CIFAR10':
        dataset = 'CIFAR-10'
    c.start('Exp. ' + str(int(opt.experiment)-1) + " (" + dataset + ");" + " Alpha="+str(opt.alpha))
