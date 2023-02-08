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
import statistics as st
import math
import numpy as np

from .t_distribution import T_Distribution
import scipy.stats as stt

def t_distribution_test(x, confidence=0.95):
    n = len(x)
    decimals = 1
    mean = round(st.mean(x), decimals)
    liberty_graus = n
    s = st.stdev(x)
    alfa = 1 - confidence
    column = 1 - alfa / 2
    t_value = T_Distribution().find_t_distribution(column, liberty_graus)
    average_variation = round((t_value * (s / math.pow(n, 1 / 2))), decimals)
    average_variation = str(average_variation)
    while len(average_variation) < decimals + 3:
        average_variation = "0" + average_variation

    mean = str(mean)
    while len(mean) < decimals + 3:
        mean = "0" + mean

    ic = stt.t.interval(alpha=0.95, df=len(x) - 1, loc=np.mean(x), scale=stt.sem(x))
    l = round(ic[0], decimals)
    r = round(ic[1], decimals)

    return str(mean) + u"\u00B1" + average_variation

class NonIid:
    def __init__(self, num_clients, aggregation_method, perc_of_clients, non_iid, model_name, strategy_name_list, dataset_name, new_clients,new_clients_train, experiment, comment, epochs):
        self.n_clients = num_clients
        self.aggregation_method = aggregation_method
        self.perc_of_clients = perc_of_clients
        self.poc_list = [0.1, 0.2, 0.3]
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

    def _get_strategy_config(self, strategy_name, poc):
        if self.aggregation_method == 'POC':
            strategy_config = f"{strategy_name}-{self.aggregation_method}-{poc}"

        # elif self.aggregation_method == 'FedLTA':
        #     strategy_config = f"{self.strategy_name}-{self.aggregation_method}-{self.decay_factor}"

        elif self.aggregation_method == 'None':
            strategy_config = f"{strategy_name}-{self.aggregation_method}"

        print("antes: ", self.aggregation_method, strategy_config)

        return strategy_config

    def start(self, ax, title):

        self.base_dir = """analysis/output/experiment_{}/{}/new_clients_{}_train_{}/{}_clients/{}/{}/{}_local_epochs/{}/""".format(
                                                                                                                            self.experiment,
                                                                                                                            self.aggregation_method + str(self.perc_of_clients),
                                                                                                                            self.new_clients,
                                                                                                                            self.new_clients_train,
                                                                                                                            self.n_clients,
                                                                                                                            self.dataset_name,
                                                                                                                            self.model_name,
                                                                                                                            self.epochs,
                                                                                                                            self.comment)

        if not Path(self.base_dir).exists():
            os.makedirs(self.base_dir+"csv")
            os.makedirs(self.base_dir + "png/")
            os.makedirs(self.base_dir + "svg/")

        models_directories = {self.strategy_name_list[i]:
                                  {j: """{}/{}/new_clients_{}_train_{}/{}/{}/{}/{}_local_epochs/""".
                              format('/home/claudio/Documentos/pycharm_projects/FedLTA/logs',
                                     self._get_strategy_config(self.strategy_name_list[i], j),
                                     self.new_clients,
                                     self.new_clients_train,
                                     self.n_clients,
                                     self.model_name,
                                     self.dataset_name,
                                     self.epochs) for j in self.poc_list} for i in range(len(self.strategy_name_list))}

        # read datasets
        print(models_directories)
        for i in range(len(self.strategy_name_list)):
            strategy_name = self.strategy_name_list[i]

            for j in self.base_files_names:
                file_name = self.base_files_names[j]
                for k in self.poc_list:
                    df = pd.read_csv(models_directories[strategy_name][k]+file_name)
                    df['Strategy'] = np.array([strategy_name]*len(df))
                    df['POC'] = np.array([k] * len(df))
                    if i == 0:
                        self.df_files_names[j] = df
                    else:
                        self.df_files_names[j] = pd.concat([self.df_files_names[j], df], ignore_index=True)

        self.server_analysis()
        df = self.evaluate_client_analysis(ax, title)
        # self.export_reports(df=df, solution_column='Strategy', poc_column='POC', index='Dataset')

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

    def evaluate_client_analysis(self, ax, title):
        # acc
        df = self.df_files_names['evaluate_client']
        df['Accuracy (%)'] = df['Accuracy'] * 100
        df['Accuracy (%)'] = df['Accuracy (%)'].round(4)
        # x_column = 'Round'
        # y_column = 'Accuracy (%)'
        # hue = 'Strategy'
        # line_plot(df=df,
        #           base_dir=self.base_dir,
        #           file_name="evaluate_client_acc_round_lineplot",
        #           x_column=x_column,
        #           y_column=y_column,
        #           title=title,
        #           hue=hue,
        #           ax=ax)

        # loss
        # x_column = 'Round'
        # y_column = 'Loss'
        # hue = 'Strategy'
        # title = ""
        # line_plot(df=df,
        #           base_dir=self.base_dir,
        #           file_name="evaluate_client_loss_round_lineplot",
        #           x_column=x_column,
        #           y_column=y_column,
        #           title=title,
        #           hue=hue)
        #
        # # size of parameters
        # print(df)
        # def strategy(df):
        #     parameters = int(df['Size of parameters'].mean())
        #
        #     return pd.DataFrame({'Size of parameters (bytes)': [parameters]})
        # df_test = df[['Round', 'Size of parameters', 'Strategy']].groupby('Strategy').apply(lambda e: strategy(e)).reset_index()[['Size of parameters (bytes)', 'Strategy']]
        # df_test.to_csv(self.base_dir+"csv/evaluate_client_size_of_parameters_round.csv", index=False)
        # print(df_test)
        # x_column = 'Strategy'
        # y_column = 'Size of parameters (bytes)'
        # hue = None
        # title = "Two layers"
        # bar_plot(df=df_test,
        #           base_dir=self.base_dir,
        #           file_name="evaluate_client_size_of_parameters_round_barplot",
        #           x_column=x_column,
        #           y_column=y_column,
        #           title=title,
        #           hue=hue,
        #          sci=True)
        # bar_plot(df=df_test,
        #          base_dir=self.base_dir,
        #          file_name="evaluate_client_size_of_parameters_round_barplot",
        #          x_column=x_column,
        #          y_column=y_column,
        #          title=title,
        #          hue=hue,
        #          log_scale=True,
        #          sci=True)

        return df





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

    def export_reports(self, df, solution_column, poc_column, index):

        model_report = {i: {} for i in df[poc_column].unique().tolist()}

        print(model_report)
        columns = 3
        index = [np.array(['Accuracy (%)']) * 3]
        models_dict = {}
        for model_name in model_report:

            report = model_report[model_name]
            accuracy = report['Accuracy (%)']*100
            accuracy_means = {}
            for column in columns:
                accuracy_means[column] = t_distribution_test(accuracy[column].tolist())

            model_metrics = []

            for column in columns:
                model_metrics.append(accuracy_means[column])

            models_dict[model_name] = model_metrics

        print("dddd")
        print(len(models_dict['FedPredict']))
        print(len(models_dict['FedAvg']))
        print(len(models_dict['FedPer']))
        print(len(models_dict['FedClassAvg']))
        df = pd.DataFrame(models_dict, index=index).round(4)

        print(df)

        output_dir = "analysis/output/performance_plots/" + dataset_type_dir + category_type_dir
        output = output_dir + str(n_splits) + "_folds/" + str(n_replications) + "_replications/"
        Path(output).mkdir(parents=True, exist_ok=True)
        # writer = pd.ExcelWriter(output + 'metrics.xlsx', engine='xlsxwriter')
        #
        # df.to_excel(writer, sheet_name='Sheet1')
        #
        # # Close the Pandas Excel writer and output the Excel file.
        # writer.save()

        max_values = self.idmax(df)
        print("zzz", max_values)
        max_columns = {'mfa': [], 'stf': [], 'map': [], 'serm': [], 'next': [], 'garg': []}

        for max_value in max_values:
            row_index = max_value[0]
            column = max_value[1]
            column_values = df[column].tolist()
            column_values[row_index] = "textbf{" + str(column_values[row_index]) + "}"

            df[column] = np.array(column_values)

        df.columns = ['POI-RGNN', 'STF-RNN', 'MAP', 'SERM', 'MHA+PE', 'GARG']

        # get improvements
        poi_rgnn = [float(i.replace("textbf{", "").replace("}", "")[:4]) for i in df['POI-RGNN'].to_numpy()]
        stf = [float(i.replace("textbf{", "").replace("}", "")[:4]) for i in df['STF-RNN'].to_numpy()]
        map = [float(i.replace("textbf{", "").replace("}", "")[:4]) for i in df['MAP'].to_numpy()]
        serm = [float(i.replace("textbf{", "").replace("}", "")[:4]) for i in df['SERM'].to_numpy()]
        mha = [float(i.replace("textbf{", "").replace("}", "")[:4]) for i in df['MHA+PE'].to_numpy()]
        garg = [float(i.replace("textbf{", "").replace("}", "")[:4]) for i in df['GARG'].to_numpy()]
        difference = []

        init = {'gowalla': 14, 'users_steps': 20}
        for i in range(init[dataset_name], len(poi_rgnn)):
            min_ = max([stf[i], map[i], serm[i], mha[i], garg[i]])
            max_ = min([stf[i], map[i], serm[i], mha[i], garg[i]])
            value = poi_rgnn[i]
            if min_ < value:
                min_ = value - min_
            else:
                min_ = 0
            if max_ < value:
                max_ = value - max_
            else:
                max_ = 0

            s = str(round(min_, 1)) + "\%--" + str(round(max_, 1)) + "\%"
            difference.append(
                [round(value, 1), round(stf[i], 1), round(map[i], 1), round(serm[i], 1), round(mha[i], 1), round(garg[i], 1), round(min_, 1), round(max_, 1), s])

        difference_df = pd.DataFrame(difference, columns=['base', 'stf', 'map', 'serm', 'mha', 'garg', 'min', 'max', 'texto'])

        difference_df.to_csv(output + "difference.csv", index=False)


        latex = df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF1-score")
        pd.DataFrame({'latex': [latex]}).to_csv(output + "latex.txt", header=False, index=False)



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

    strategy_name_list = ['FedPredict', 'FedAVG', 'FedClassAvg', 'FedPer']

    fig, axs = plt.subplots(2, 2, figsize=(12,9))
    n_clients = 50
    aggregation_method = 'POC'
    poc = 0.2
    non_iid = True
    comment = ''
    epochs = 1
    model = 'DNN'
    dataset = 'MNIST'
    # Experiment 1
    new_clients = False
    new_clients_train = False
    experiment = 1
    c = NonIid(n_clients, aggregation_method, poc, non_iid, model, strategy_name_list, dataset, new_clients, new_clients_train, experiment, comment, epochs)
    c.start(ax=axs[0, 0], title='Exp. ' + str(experiment))
    # Experiment 2
    new_clients = True
    new_clients_train = False
    experiment = 2
    c = NonIid(n_clients, aggregation_method, poc, non_iid, model, strategy_name_list, dataset, new_clients,
               new_clients_train, experiment, comment, epochs)
    c.start(ax=axs[0, 1], title='Exp. ' + str(experiment))
    axs[0, 1].get_legend().remove()
    # Experiment 3
    new_clients = True
    new_clients_train = True
    experiment = 3
    c = NonIid(n_clients, aggregation_method, poc, non_iid, model, strategy_name_list, dataset, new_clients,
               new_clients_train, experiment, comment, epochs)
    c.start(ax=axs[1, 0], title='Exp. ' + str(experiment))
    axs[1, 0].get_legend().remove()
    # Experiment 4
    new_clients = True
    new_clients_train = True
    experiment = 4
    c = NonIid(n_clients, aggregation_method, poc, non_iid, model, strategy_name_list, dataset, new_clients,
               new_clients_train, experiment, comment, epochs)
    print(c.n_clients, " ", c.strategy_name_list)
    c.start(ax=axs[1, 1], title='Exp. ' + str(experiment))
    axs[1, 1].get_legend().remove()

    fig.savefig("joint_plot.png", bbox_inches='tight', dpi=400)
    fig.savefig("joint_plot.svg", bbox_inches='tight', dpi=400)