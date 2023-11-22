import copy
from pathlib import Path

import numpy as np
import pandas as pd
from base_plots import bar_plot, line_plot, ecdf_plot, box_plot, violin_plot
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import os

class Varying_Shared_layers:

    def __init__(self, tp, strategy_name, new_clients, aggregation_method, fraction_fit, new_clients_train, num_clients, model_name, dataset,
                 class_per_client, alpha, num_rounds, epochs, comment, compression):

        self.type = tp
        self.strategy_name = strategy_name
        self.aggregation_method = aggregation_method
        self.fraction_fit = fraction_fit
        self.new_clients = new_clients
        self.new_clients_train = new_clients_train
        self.num_clients = num_clients
        self.model_name = model_name
        self.dataset = dataset
        self.class_per_client = class_per_client
        self.alpha = alpha
        self.num_rounds = num_rounds
        self.epochs = epochs
        self.comment = comment
        self.compression = compression
        self.model_name_list = [model.replace("CNN_2", "CNN-a").replace("CNN_3", "CNN-b").replace("CNN_1", "CNN-a") for model in self.model_name]
        self.dataset_name_list = [dataset.replace("CIFAR10", "CIFAR-10") for dataset in self.dataset]
        if "dls_compredict" in self.compression:
            self.experiment = "dls_compredict"
        elif "dls" in self.compression:
            self.experiment = "dls"
        elif "compredict" in self.compression:
            self.experiment = "compredict"
        elif "sparsification" in self.compression:
            self.experiment = "sparsification"

    def parameters_reduction(self):

        df = self.df_concat
        df_aux = copy.deepcopy(df)

        print("e1: ", df.query("Solution=='$FedAvg+FP$' and Model=='CNN-b' and Dataset=='EMNIST'")[
            'Size of parameters (MB)'])
        print("e2: ", df.query("Solution=='$FedAvg+FP_{s}$' and Model=='CNN-b' and Dataset=='EMNIST'")[[
            'Size of parameters (MB)']])

        def comparison_with_shared_layers(df):

            round = int(df['Round'].values[0])
            dataset = str(df['Dataset'].values[0])
            alpha = float(df['\u03B1'].values[0])
            model = str(df['Model'].values[0])
            solution = df['Solution'].values[0]
            # print("interes: ", round, dataset, alpha, model)
            df_copy = copy.deepcopy(df_aux.query("""Round == {} and Dataset == '{}' and \u03B1 == {} and Model == '{}'""".format(round, dataset, alpha, model)))
            # print("apos: ", df_copy.columns)
            target = df_copy[df_copy['Solution'] == "$FedAvg+FP$"]
            target_acc = target['Accuracy (%)'].mean()
            target_size = target['Size of parameters (MB)'].mean()

            if len(target) != len(df):
                print("oioi", len(target), len(df), "co: ", round, dataset, alpha, model, "\n", df_aux.query("""Dataset == '{}' and \u03B1 == {} and Model == '{}'""".format(dataset, alpha, model))[['Dataset', 'Round', 'Solution', 'Strategy', '\u03B1', 'Model']].drop_duplicates())
                exit()

            acc = df['Accuracy (%)'].mean()
            accuracy = df['Accuracy (%)'].mean()
            size = df['Size of parameters (MB)'].mean()
            acc_reduction = target_acc - acc
            size_reduction = (target_size - size)
            # if model=='CNN-b' and dataset=='EMNIST' and solution =='$FedAvg+FP_{s}$':
            #     print("di", target_size, size, size_reduction)
            #     print(df['Size of parameters (MB)'])
            #     print(df[df['Solution']=='$FedAvg+FP_{s}$'])
            size_reduction_percentage = (1 - size/target_size) * 100
            # acc_weight = 1
            # size_weight = 1
            # acc_score = acc_score *acc_weight
            # size_reduction = size_reduction * size_weight
            # score = 2*(acc_score * size_reduction)/(acc_score + size_reduction)
            # if df['Solution'].tolist()[0] == "{1, 2, 3, 4}":
            #     acc_reduction = 0.0001
            #     size_reduction = 0.0001
            # if df['Solution'].unique().tolist()[0] == "$FedAvg+FP_{s}$":
            #     print("tamanho: ", len(target['Size of parameters (MB)'].tolist()))
            #     print('Parameters reduction (%)', dataset, model, alpha, size_reduction_percentage, size, target_size)


            return pd.DataFrame({'Accuracy reduction (%)': [acc_reduction], 'Parameters reduction (MB)': [size_reduction],
                                 'Parameters reduction (%)': [size_reduction_percentage], 'Accuracy (%)': [accuracy], 'Size of parameters (MB)': [size]})

        df = df[['Accuracy (%)', 'Size of parameters (MB)', 'Strategy', 'Solution', 'Round', '\u03B1', 'Dataset',
                 'Model']].groupby(
            by=['Strategy', 'Round', 'Solution', 'Dataset', '\u03B1', 'Model']).apply(
            lambda e: comparison_with_shared_layers(df=e)).reset_index()
        # [
        #     ['Strategy', 'Round', 'Solution', '\u03B1', 'Accuracy (%)', 'Accuracy reduction (%)',
        #      'Parameters reduction (MB)', 'Size of parameters (MB)', 'Dataset', 'Model']]

        print("e11: ", df.query("Solution=='$FedAvg+FP$' and Model=='CNN-b' and Dataset=='EMNIST'")[
                'Size of parameters (MB)'])
        print("e12: ", df.query("Solution=='$FedAvg+FP_{s}$' and Model=='CNN-b' and Dataset=='EMNIST'")[[
                'Size of parameters (MB)', 'Parameters reduction (MB)']])

        self.df_concat = df


    def start(self):

        self.build_filenames()



        self.parameters_reduction()

        self.df_concat = self.df_concat[['Strategy', 'Round', 'Solution', 'Dataset', 'α', 'Model', 'level_6',
       'Accuracy reduction (%)', 'Parameters reduction (MB)',
       'Parameters reduction (%)', 'Accuracy (%)', 'Size of parameters (MB)']]

        # df_concat = None

        # for model in self.model_name:
        #     for dataset in self.dataset:
        #         model_name = model.replace("CNN_2", "CNN-a").replace("CNN_3", "CNN-b")
        #         dataset_name = dataset.replace("CIFAR10", "CIFAR-10")
        #         self.evaluate_client_analysis_shared_layers(model_name, dataset_name)

        # self.evaluate_client_analysis_differnt_models(df_concat)
        # print(df_concat['Strategy'].unique().tolist())
        # exit()
        self.evaluate_client_joint_parameter_reduction(self.df_concat)
        alphas = self.df_concat['\u03B1'].unique().tolist()
        models = self.df_concat['Model'].unique().tolist()
        # print(self.df_concat.columns)
        # print(self.df_concat.isna())
        self.df_concat = self.build_filename_fedavg(self.df_concat)
        # print(self.df_concat.columns)
        # print(self.df_concat.isna())
        # exit()

        for alpha in alphas:
            self.evaluate_client_joint_accuracy(self.df_concat, alpha)
            self.joint_table(self.df_concat, alpha=alpha, models=models)
            self.joint_table(self.df_concat, alpha=alpha, models=models, target_col='Size of parameters (MB)')

        # for alpha in alphas:
        #     self.evaluate_client_joint_accuracy(df_concat, alpha)
        self.similarity()
        # for alpha in alphas:
        #     self.evaluate_client_norm_analysis_nt(alpha)
        # self.evaluate_client_norm_analysis()

    def convert_shared_layers(self, df):

        shared_layers_list = df['Solution'].tolist()
        for i in range(len(shared_layers_list)):
            shared_layer = str(shared_layers_list[i])
            if "dls_compredict" == shared_layer:
                shared_layers_list[i] = "$FedAvg+FP_{dc}$"
            elif "dls" == shared_layer:
                shared_layers_list[i] = "$FedAvg+FP_{d}$"

            elif "compredict" == shared_layer:
                shared_layers_list[i] = "$FedAvg+FP_{c}$"
            elif "fedkd" == shared_layer:
                shared_layers_list[i] = "$FedAvg+FP_{kd}$"
            elif "sparsification" == shared_layer:
                shared_layers_list[i] = "$FedAvg+FP_{s}$"
            elif "per" == shared_layer:
                shared_layers_list[i] = "$FedAvg+FP_{per}$"
            elif "no" == shared_layer:
                shared_layers_list[i] = "$FedAvg+FP$"
            # new_shared_layer = "{"
            # for layer in shared_layer:
            #     if len(new_shared_layer) == 1:
            #         new_shared_layer += layer
            #     else:
            #         new_shared_layer += ", " + layer
            #
            # new_shared_layer += "}"
            # if shared_layer == "no":
            #     new_shared_layer = "$FedAvg+FP$"
            # if shared_layer == "50":
            #     new_shared_layer = "50% of the layers"
            #
            # shared_layers_list[i] = new_shared_layer

        df['Solution'] = np.array(shared_layers_list)
        print(df['Solution'].unique().tolist())

        return df

    def build_filenames(self):

        # files = ["evaluate_client.csv", "similarity_between_layers.csv"]
        files = ["evaluate_client.csv", "similarity_between_layers.csv"]
        df_concat = None
        df_concat_similarity = None
        df_concat_norm = None
        for file in files:
            for compression in self.compression:
                for a in self.alpha:
                    for model in self.model_name:
                        for dataset in self.dataset:



                            filename = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/FL-H.IAAC/" + f"logs/{self.type}/{self.strategy_name}-{self.aggregation_method}-{self.fraction_fit}/new_clients_{self.new_clients}_train_{self.new_clients_train}/{self.num_clients}/{model}/{dataset}/classes_per_client_{self.class_per_client}/alpha_{a}/{self.num_rounds}_rounds/{self.epochs}_local_epochs/{self.comment}_comment/{str(compression)}_compression/{file}"
                            # if "/home/claudio/Documentos/pycharm_projects/FL-H.IAAC/logs/torch/FedPredict-None-0.3/new_clients_False_train_False/20/CNN_3/EMNIST/classes_per_client_2/alpha_5.0/100_rounds/1_local_epochs/set_comment/no_compression" not in filename:
                            #     continue
                            try:
                                df = pd.read_csv(filename).dropna()
                            except:
                                print("arquivo inexistente")
                                print(filename)
                                continue

                            # model_name = model.replace("CNN_2", "CNN-a").replace("CNN_3", "CNN-b")
                            # dataset_name = dataset.replace("CIFAR10", "CIFAR-10")
                            print("leu: ", filename)
                            model_name = model.replace("CNN_2", "CNN-a").replace("CNN_3", "CNN-b")
                            dataset_name = dataset.replace("CIFAR10", "CIFAR-10")
                            if "evaluate" in file:
                                df['Solution'] = np.array([compression] * len(df))
                                df['Strategy'] = np.array([self.strategy_name] * len(df))
                                df['\u03B1'] = np.array([a]*len(df))
                                df['Model'] = np.array([model_name]*len(df))
                                df['Dataset'] = np.array([dataset_name]*len(df))
                                df['Accuracy (%)'] = df['Accuracy'].to_numpy() * 100
                                df['Size of parameters (MB)'] = df['Size of parameters'].to_numpy() / 1000000
                                df['Accuracy (%)'] = df['Accuracy (%)'].round(4)
                                # df['Round'] = np.array(df['Server round'].tolist())

                                if df_concat is None:
                                    df_concat = df
                                else:
                                    df_concat = pd.concat([df_concat, df], ignore_index=True)
                            elif "similarity" in file:
                                if compression not in ["dls", "dls_compredict", "sparsification", "no"]:
                                    continue
                                df['\u03B1'] = np.array([a] * len(df))
                                df['Round'] = np.array(df['Server round'].tolist())
                                df['Dataset'] = np.array([dataset_name] * len(df))
                                df['Model'] = np.array([model_name] * len(df))
                                df['Solution'] = np.array([compression] * len(df))
                                df['Strategy'] = np.array([self.strategy_name] * len(df))
                                if df_concat_similarity is None:
                                    df_concat_similarity = df
                                else:
                                    df_concat_similarity = pd.concat([df_concat_similarity, df], ignore_index=True)
                            # elif "norm" in file:
                            #     if layers not in ["dls", "dls_compredict", "sparsification"]:
                            #         continue
                            #     df['\u03B1'] = np.array([a] * len(df))
                            #     df['Server round'] = np.array(df['Round'].tolist())
                            #     df['Dataset'] = np.array([dataset_name] * len(df))
                            #     df['Model'] = np.array([model_name] * len(df))
                            #     df['Solution'] = np.array([layers] * len(df))
                            #     df['Strategy'] = np.array([self.strategy_name] * len(df))
                            #     if df_concat_norm is None:
                            #         df_concat_norm = df
                            #     else:
                            #         df_concat_norm = pd.concat([df_concat_norm, df], ignore_index=True)

                            # if "dls_compredict" in layers and file == 'evaluate_client.csv':
                            #     print(df)

        self.df_concat = self.convert_shared_layers(df_concat)
        print("contruído: ", df_concat.columns)
        # print(df_concat.isna())
        # exit()
        # print("e1: ", self.df_concat.query("Solution=='$FedAvg+FP$' and Model=='CNN-b' and Dataset=='EMNIST'")[
        #     'Size of parameters (MB)'])
        # print("e2: ", self.df_concat.query("Solution=='$FedAvg+FP_{s}$' and Model=='CNN-b' and Dataset=='EMNIST'")[
        #     'Size of parameters (MB)'])
        # exit()
        # exit()
        self.df_concat_similarity = df_concat_similarity
        self.df_concat_norm = df_concat_norm
        # print("Leu similaridade", df_concat_similarity[['Round', '\u03B1', 'Similarity', 'Dataset', 'Model', 'Layer']].drop_duplicates().to_string())
        # exit()

    def build_filename_fedavg(self, df_concat, use_mean=True):

        files = ["evaluate_client.csv"]
        df_concat_similarity = None
        for layers in ["dls"]:
            for a in self.alpha:
                for model in self.model_name:
                    for dataset in self.dataset:
                        for file in files:

                            filename = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/FL-H.IAAC/" + f"logs/{self.type}/FedAVG-{self.aggregation_method}-{self.fraction_fit}/new_clients_{self.new_clients}_train_{self.new_clients_train}/{self.num_clients}/{model}/{dataset}/classes_per_client_{self.class_per_client}/alpha_{a}/{self.num_rounds}_rounds/{self.epochs}_local_epochs/{self.comment}_comment/{layers}_compression/{file}"
                            if not os.path.exists(filename):
                                print("não achou fedavg")
                                continue
                            df = pd.read_csv(filename).dropna()
                            model_name = model.replace("CNN_2", "CNN-a").replace("CNN_3", "CNN-b")
                            dataset_name = dataset.replace("CIFAR10", "CIFAR-10")
                            print("la: ", a, model, dataset, file)
                            print(df.isnull())
                            if "evaluate" in file:
                                df['Solution'] = np.array([layers] * len(df))
                                df['Strategy'] = np.array(['FedAvg'] * len(df))
                                df['Solution'] = np.array(["$FedAvg$"] * len(df))
                                df['\u03B1'] = np.array([a]*len(df))
                                df['Model'] = np.array([model_name]*len(df))
                                df['Dataset'] = np.array([dataset_name]*len(df))
                                df['Accuracy (%)'] = df['Accuracy'].to_numpy() * 100
                                df['Size of parameters (MB)'] = df['Size of parameters'].to_numpy() / 1000000
                                df['Accuracy reduction (%)'] = np.array([0]*len(df))
                                df['Parameters reduction (MB)'] = np.array([0]*len(df))
                                df['Parameters reduction (%)'] = np.array([0] * len(df))

                                def summary(df):

                                    acc = df['Accuracy (%)'].mean()

                                    return pd.DataFrame({'Accuracy (%)': [acc]})

                                if use_mean:
                                    df = df.groupby(
                                        ['Dataset', 'Model', '\u03B1', 'Strategy', 'Solution', 'Round']).mean().reset_index()

                                if df_concat is None:
                                    df_concat = df
                                else:
                                    print("concat: ", df_concat.columns)
                                    print("df: ", df.columns)
                                    df = df[['Strategy', 'Round', 'Solution', 'Dataset', 'α', 'Model',
       'Accuracy reduction (%)', 'Parameters reduction (MB)',
       'Parameters reduction (%)', 'Accuracy (%)', 'Size of parameters (MB)']]
                                    df_concat = pd.concat([df, df_concat], ignore_index=True)

        return df_concat

    def evaluate_client_norm_analysis_nt(self, alpha):

        df = self.df_concat_norm.query("""\u03B1 == {}""".format(alpha))
        unique_nt_filter = [1, 4, 8, 12]
        print("filtro: ", unique_nt_filter)
        nt = df['nt'].tolist()
        for i in range(len(nt)):
            if nt[i] not in unique_nt_filter:
                nt[i] = -1
        df['nt'] = np.array(nt)
        df = df[df['nt'] != -1]
        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/""".format(
            self.experiment, str(self.dataset), self.num_clients, self.num_rounds, self.fraction_fit,
            str(self.model_name),
            str(self.alpha), self.comment)

        if len(self.dataset) == 2 and len(self.model_name) == 2 and "dls_compredict" in self.compression:
            fig, ax = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(6, 6))
            y_max = 2
            x_column = 'Round'
            y_column = 'Norm'
            hue = 'nt'

            model_name_index = 0
            dataset_name_index = 0
            title = """{}; {}""".format(self.model_name_list[model_name_index], self.dataset_name_list[dataset_name_index])
            line_plot(
                        ax=ax[model_name_index, dataset_name_index],
                        df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[dataset_name_index], self.model_name_list[model_name_index])),
                      base_dir=base_dir,
                      file_name="",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      type=1,
                      y_lim=True,
                      y_max=y_max,
                      y_min=0)
            ax[model_name_index, dataset_name_index].get_legend().remove()
            ax[model_name_index, dataset_name_index].set_xlabel('')
            ax[model_name_index, dataset_name_index].set_ylabel('')

            model_name_index = 0
            dataset_name_index = 1
            title = """{}; {}""".format(self.model_name_list[model_name_index], self.dataset_name_list[dataset_name_index])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[dataset_name_index], self.model_name_list[model_name_index])),
                base_dir=base_dir,
                file_name="",
                x_column=x_column,
                y_column=y_column,
                title=title,
                hue=hue,
                type=1,
                y_lim=True,
                y_max=y_max,
                y_min=0)
            ax[model_name_index, dataset_name_index].get_legend().remove()
            ax[model_name_index, dataset_name_index].set_xlabel('')
            ax[model_name_index, dataset_name_index].set_ylabel('')

            model_name_index = 1
            dataset_name_index = 0
            title = """{}; {}""".format(self.model_name_list[model_name_index], self.dataset_name_list[dataset_name_index])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[dataset_name_index], self.model_name_list[model_name_index])),
                base_dir=base_dir,
                file_name="",
                x_column=x_column,
                y_column=y_column,
                title=title,
                hue=hue,
                type=1,
                y_lim=True,
                y_max=y_max,
                y_min=0)
            ax[model_name_index, dataset_name_index].get_legend().remove()
            ax[model_name_index, dataset_name_index].set_xlabel('')
            ax[model_name_index, dataset_name_index].set_ylabel('')

            model_name_index = 1
            dataset_name_index = 1
            title = """{}; {}""".format(self.model_name_list[model_name_index], self.dataset_name_list[dataset_name_index])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[dataset_name_index], self.model_name_list[model_name_index])),
                base_dir=base_dir,
                file_name="",
                x_column=x_column,
                y_column=y_column,
                title=title,
                hue=hue,
                type=1,
                y_lim=True,
                y_max=y_max,
                y_min=0)
            ax[model_name_index, dataset_name_index].get_legend().remove()
            ax[model_name_index, dataset_name_index].set_xlabel('')
            ax[model_name_index, dataset_name_index].set_ylabel('')

            fig.suptitle("", fontsize=16)
            fig.supxlabel(x_column, y=-0.02)
            fig.supylabel(y_column, x=-0.005)
            # plt.tight_layout(pad=0.5)

            plt.subplots_adjust(wspace=0.07, hspace=0.14)
            lines_labels = [ax[0, 0].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines, labels, title="""nt""".format(alpha), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.5))
            figure = fig.get_figure()
            Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
            Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
            filename = """norm_nt_round_{}_{}_alpha_{}""".format(
                self.experiment, str(self.dataset), alpha)
            figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
            figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)

    def evaluate_client_norm_analysis(self):

        df = self.df_concat_norm
        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/""".format(
            self.experiment, str(self.dataset), self.num_clients, self.num_rounds, self.fraction_fit,
            str(self.model_name),
            str(self.alpha), self.comment)

        if len(self.dataset) == 2 and len(self.model_name) == 2 and "dls_compredict" in self.compression:
            fig, ax = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(6, 6))
            y_max = 1
            x_column = 'Round'
            y_column = 'Norm'
            hue = '\u03B1'
            hue_order = df[hue].unique().tolist().sort(reverse=False)

            model_name_index = 0
            dataset_name_index = 0
            title = """{}; {}""".format(self.model_name_list[model_name_index], self.dataset_name_list[dataset_name_index])
            print("agrupou")
            print("colunas: ", df.columns)
            print(df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[dataset_name_index],
                                                                           self.model_name_list[model_name_index])).groupby(["Round", "\u03B1"]).mean().reset_index()[[x_column, y_column, hue]])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[dataset_name_index],
                                                                           self.model_name_list[model_name_index])).groupby(["Round", "\u03B1"]).mean().reset_index()[[x_column, y_column, hue]],
                base_dir=base_dir,
                file_name="teste",
                x_column=x_column,
                y_column=y_column,
                hue=hue,
                hue_order=hue_order,
                style=hue_order,
                title=title,
                type=1,
                y_lim=True,
                y_max=y_max,
                y_min=0)
            # exit()
            ax[model_name_index, dataset_name_index].get_legend().remove()
            ax[model_name_index, dataset_name_index].set_xlabel('')
            ax[model_name_index, dataset_name_index].set_ylabel('')

            model_name_index = 0
            dataset_name_index = 1
            title = """{}; {}""".format(self.model_name_list[model_name_index], self.dataset_name_list[dataset_name_index])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[dataset_name_index],
                                                                           self.model_name_list[model_name_index])).groupby(
                    ["Round", "\u03B1"]).mean().reset_index()[[x_column, y_column, hue]],
                base_dir=base_dir,
                file_name="teste",
                x_column=x_column,
                y_column=y_column,
                hue=hue,
                hue_order=hue_order,
                style=hue_order,
                title=title,
                type=1,
                y_lim=True,
                y_max=y_max,
                y_min=0)
            ax[model_name_index, dataset_name_index].get_legend().remove()
            ax[model_name_index, dataset_name_index].set_xlabel('')
            ax[model_name_index, dataset_name_index].set_ylabel('')

            model_name_index = 1
            dataset_name_index = 0
            title = """{}; {}""".format(self.model_name_list[model_name_index], self.dataset_name_list[dataset_name_index])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[dataset_name_index],
                                                                           self.model_name_list[model_name_index])).groupby(
                    ["Round", "\u03B1"]).mean().reset_index()[[x_column, y_column, hue]],
                base_dir=base_dir,
                file_name="teste",
                x_column=x_column,
                y_column=y_column,
                hue=hue,
                hue_order=hue_order,
                style=hue_order,
                title=title,
                type=1,
                y_lim=True,
                y_max=y_max,
                y_min=0)
            ax[model_name_index, dataset_name_index].get_legend().remove()
            ax[model_name_index, dataset_name_index].set_xlabel('')
            ax[model_name_index, dataset_name_index].set_ylabel('')

            model_name_index = 1
            dataset_name_index = 1
            title = """{}; {}""".format(self.model_name_list[model_name_index], self.dataset_name_list[dataset_name_index])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[dataset_name_index],
                                                                           self.model_name_list[model_name_index])).groupby(
                    ["Round", "\u03B1"]).mean().reset_index()[[x_column, y_column, hue]],
                base_dir=base_dir,
                file_name="teste",
                x_column=x_column,
                y_column=y_column,
                hue=hue,
                hue_order=hue_order,
                style=hue_order,
                title=title,
                type=1,
                y_lim=True,
                y_max=y_max,
                y_min=0)
            ax[model_name_index, dataset_name_index].get_legend().remove()
            ax[model_name_index, dataset_name_index].set_xlabel('')
            ax[model_name_index, dataset_name_index].set_ylabel('')

            fig.suptitle("", fontsize=16)
            fig.supxlabel(x_column, y=-0.02)
            fig.supylabel(y_column, x=-0.005)
            # plt.tight_layout(pad=0.5)

            plt.subplots_adjust(wspace=0.07, hspace=0.14)
            lines_labels = [ax[0, 0].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines, labels, title="""\u03B1""".format(), loc='upper center', ncol=4,
                       bbox_to_anchor=(0.5, 1.06))
            figure = fig.get_figure()
            Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
            Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
            filename = """norm_round_{}_{}_alpha_{}""".format(
                self.experiment, str(self.dataset), alpha)
            figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
            figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)


    def evaluate_client_joint_parameter_reduction(self, df):

        # df = df[df['Solution'] == "$FedAvg+FP_d$"]

        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/""".format(
            self.experiment, str(self.dataset), self.num_clients, self.num_rounds, self.fraction_fit,
            str(self.model_name),
            str(self.alpha), self.comment)

        df['Parameters saving (%)'] = np.array(df['Parameters reduction (%)'].tolist())
        x_column = 'Round'
        y_column = 'Parameters saving (%)'
        hue = 'Solution'
        style = '\u03B1'
        y_min = 0
        if self.experiment == "dls_compredict":
            compression = ["$FedAvg+FP_{dc}$", "$FedAvg+FP_{d}$", "$FedAvg+FP_{c}$", "$FedAvg+FP_{kd}$", "$FedAvg+FP_{s}$", "$FedAvg+FP_{per}$"]
        elif "dls" in self.compression:
            compression = ["$FedAvg+FP_{d}$", 'FedPredict', 'FedAvg']
        else:
            compression = ["$FedAvg+FP_{c}$", 'FedPredict', 'FedAvg']

        if "$FedAvg+FP_{dc}$" not in df['Solution'].tolist():
            y_max = 80
        else:
            y_max = 100

        if len(self.dataset) >= 2:
            fig, ax = plt.subplots(3, 2, sharex='all', sharey='all', figsize=(6, 6))
            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[0])
            x = df[x_column].tolist()
            y = df[y_column].tolist()
            print("jointplot: \n", df.query("""Dataset == '{}'""".format(self.dataset_name_list[0])))
            line_plot(ax=ax[0, 0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[0])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=y_max,
                      y_min=y_min,
                      n=1)

            ax[0, 0].get_legend().remove()
            ax[0, 0].set_xlabel('')
            ax[0, 0].set_ylabel('')
            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[1])
            line_plot(ax=ax[0, 1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[1])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=y_max,
                      y_min=y_min,
                      n=1)

            ax[0, 1].get_legend().remove()
            ax[0, 1].set_xlabel('')
            ax[0, 1].set_ylabel('')

            title = """{}; {}""".format(self.dataset_name_list[1], self.model_name_list[0])
            line_plot(ax=ax[1, 0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[1], self.model_name_list[0])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=y_max,
                      y_min=y_min,
                      n=1)

            ax[1, 0].get_legend().remove()
            # ax[1, 0].legend(fontsize=7, ncol=2)
            ax[1, 0].set_xlabel('')
            ax[1, 0].set_ylabel('')

            title = """{}; {}""".format(self.dataset_name_list[1], self.model_name_list[1])
            line_plot(ax=ax[1, 1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[1], self.model_name_list[1])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=y_max,
                      y_min=y_min,
                      n=1)

            ax[1, 1].get_legend().remove()
            ax[1, 1].set_xlabel('')
            ax[1, 1].set_ylabel('')

            #

            title = """{}; {}""".format(self.dataset_name_list[2], self.model_name_list[0])
            line_plot(ax=ax[2, 0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[2],
                                                                                 self.model_name_list[0])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=y_max,
                      y_min=y_min,
                      n=1)

            ax[2, 0].get_legend().remove()
            # ax[1, 0].legend(fontsize=7, ncol=2)
            ax[2, 0].set_xlabel('')
            ax[2, 0].set_ylabel('')

            title = """{}; {}""".format(self.dataset_name_list[2], self.model_name_list[1])
            line_plot(ax=ax[2, 1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[2],
                                                                                 self.model_name_list[1])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=y_max,
                      y_min=y_min,
                      n=1)

            ax[2, 1].get_legend().remove()
            ax[2, 1].set_xlabel('')
            ax[2, 1].set_ylabel('')

            fig.suptitle("", fontsize=16)
            fig.supxlabel(x_column, y=-0.02)
            fig.supylabel(y_column, x=-0.005)
            lines_labels = [ax[0, 1].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            print("linhas")
            print(lines)
            print(lines[0].get_color(), lines[0].get_ls())
            print("rotulos")
            print(labels)
            colors = []
            for i in range(len(lines)):
                color = lines[i].get_color()
                colors.append(color)
                ls = lines[i].get_ls()
                if ls not in ["o"]:
                    ls = "o"
            markers = ["-", "--"]

            f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
            n = len(compression) + 1
            handles = [f("o", colors[i]) for i in range(n)]
            new_labels = []
            for i in range(len(labels)):
                if i != n:
                    new_labels.append(labels[i])
                else:
                    print("label: ", labels[i])
            new_labels[-1] = '\u03B1=' + new_labels[-1]
            new_labels[-2] = '\u03B1=' + new_labels[-2]
            new_labels = new_labels[1:]

            handles += [plt.Line2D([], [], linestyle=markers[i], color="k") for i in range(len(markers))]
            fig.legend(handles[1:], new_labels, fontsize=9, ncols=4, bbox_to_anchor=(0.90, 1.02))
            figure = fig.get_figure()
            Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
            Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
            filename = """parameters_reduction_percentage_{}_varying_shared_layers_lineplot_joint_{}""".format(self.experiment, str(self.dataset))
            figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
            figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)

        else:
            fig, ax = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(8, 6))
            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[0])
            x = df[x_column].tolist()
            y = df[y_column].tolist()
            print("jointplot: \n", df.query("""Dataset == '{}'""".format(self.dataset_name_list[0])))
            line_plot(ax=ax[0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[0])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      # hue_order=compression_methods,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=100,
                      y_min=20,
                      n=1)

            ax[0].get_legend().remove()
            ax[0].set_xlabel('')
            ax[0].set_ylabel('')
            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[1])
            line_plot(ax=ax[1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[1])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      # hue_order=compression_methods,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=100,
                      y_min=20,
                      n=1)

            ax[1].get_legend().remove()
            ax[1].set_xlabel('')
            ax[1].set_ylabel('')

            fig.suptitle("", fontsize=16)
            fig.supxlabel(x_column, y=-0.02)
            fig.supylabel(y_column, x=-0.005)
            # plt.tight_layout(pad=0.5)

            plt.subplots_adjust(wspace=0.07, hspace=0.14)
            lines_labels = [ax[0].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            # fig.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.06))
            figure = fig.get_figure()
            Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
            Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
            filename = """parameters_reduction_percentage_{}_varying_shared_layers_lineplot_joint_{}""".format(self.experiment, str(self.dataset))
            figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
            figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)

    def t_distribution(self, data, ci):

        min_, max_ = st.t.interval(alpha=ci, df=len(data) - 1,
                      loc=np.mean(data),
                      scale=st.sem(data))

        mean = np.mean(data)
        average_variation = (mean - min_).round(1)
        mean = mean.round(1)
        if np.isnan(average_variation):
            print("nulo: ", average_variation, len(data), np.mean(data), mean, ci, min_, max_)
            average_variation = "0.0"

        return str(mean) + u"\u00B1" + str(average_variation)

    def accuracy_improvement(self, df, range_of_string, target_col):

        df_difference = copy.deepcopy(df)
        columns = df.columns.tolist()
        indexes = df.index.tolist()

        datasets = ['EMNIST', 'GTSRB']
        solutions = pd.Series([i[1] for i in indexes]).unique().tolist()
        reference_solutions = {}
        for solution_key in solutions:
            if "FP_{dc}" in solution_key or "FP_{d}" in solution_key or "FP_{c}" in solution_key or "FP_{kd}" in solution_key or "FP" in solution_key or "FP_{s}" in solution_key or "FP_{per}" in solution_key:
                reference_solutions[solution_key] = solution_key.replace("+FP_{dc}", "").replace("+FP_{d}", "").replace("+FP_{c}", "").replace("+FP_{kd}", "").replace("+FP_{s}", "").replace("+FP_{per}", "").replace("+FP", "")

        for dataset in datasets:
            for solution in reference_solutions:
                reference_index = (dataset, solution)
                target_index = (dataset, reference_solutions[solution])

                for column in columns:
                    difference = str(
                        round(float(df.loc[reference_index, column][:range_of_string]) - float(df.loc[target_index, column][:range_of_string]), 1))
                    difference = str(round(float(difference) * 100 / float(df.loc[target_index, column][:range_of_string]), 1))
                    if "Acc" in target_col:
                        if difference[0] != "-":
                            difference = "textuparrow" + difference
                        elif float(difference[1:4]) > 0:
                            difference = "textuparrow" + difference.replace("-", "")
                    else:
                        if difference[0] != "-":
                            difference = difference
                        elif float(difference[1:4]) > 0:
                            difference = difference.replace("-", "")
                    print("solucao: ", solution)
                    if solution in ["$FedAvg$", "$FedAvg+FP$"] and 'Acc' not in target_col:
                        print("foi:", difference)
                        difference = difference.replace("0.0", "")
                        print("depois:", difference)
                        df_difference.loc[reference_index, column] = difference + df.loc[
                            reference_index, column]
                    else:
                        df_difference.loc[reference_index, column] = str("(" + difference + "%)" + df.loc[
                            reference_index, column])

        print(indexes)
        print(indexes[0])
        print(df_difference)

        return df_difference

    def joint_table(self, df, alpha, models, target_col='Accuracy (%)'):

        shared_layers = df['Solution'].unique().tolist()

        model_report = {i: {} for i in df['Model'].sort_values().unique().tolist()}

        # df = df[df['Round'] == 100]
        print("receb: ", df.columns)
        df_test = df[
            ['Round', 'Size of parameters (MB)', 'Solution', 'Accuracy (%)', '\u03B1', 'Dataset', 'Model', 'Parameters reduction (%)']]

        # df_test = df_test.query("""Round in [10, 100]""")
        print("agrupou table")
        print(df_test)
        convert_dict = {0.1: 5, 0.2: 10, 0.3: 15, 0.4: 20}
        # df_test['Fraction fit'] = np.array([convert_dict[i] for i in df_test['Fraction fit'].tolist()])

        columns = shared_layers

        index = [np.array(['EMNIST'] * len(columns) + ['GTSRB'] * len(columns)), np.array(columns * 2)]

        models_dict = {}
        ci = 0.95
        for shared_layer in model_report:

            mnist_acc = {}
            cifar10_acc = {}
            for column in columns:

                # mnist_acc[column] = (self.filter(df_test, experiment, 'MNIST', float(column), strategy=model_name)['Accuracy (%)']*100).mean().round(6)
                # cifar10_acc[column] = (self.filter(df_test, experiment, 'CIFAR10', float(column), strategy=model_name)['Accuracy (%)']*100).mean().round(6)
                mnist_acc[column] = self.t_distribution((self.filter(df_test, model=shared_layer, dataset='EMNIST', alpha=alpha, shared_layer=column)[
                                         target_col]).tolist(), ci)
                cifar10_acc[column] = self.t_distribution((self.filter(df_test,model=shared_layer, dataset='GTSRB', alpha=alpha, shared_layer=column)[
                                           target_col]).tolist(), ci)

            model_metrics = []

            for column in columns:
                model_metrics.append(mnist_acc[column])
            for column in columns:
                model_metrics.append(cifar10_acc[column])

            models_dict[shared_layer] = model_metrics

        df_table = pd.DataFrame(models_dict, index=index).round(4)
        print(df_table.to_string())

        range_of_string = {'Accuracy (%)': 4, 'Size of parameters (MB)': 3}[target_col]
        df_accuracy_improvements = self.accuracy_improvement(df_table, range_of_string, target_col)
        print(df_accuracy_improvements)


        indexes = df_table.index.tolist()
        n_solutions = len(pd.Series([i[1] for i in indexes]).unique().tolist()) + 1

        max_values = self.idmax(df_table, n_solutions, range_of_string, target_col)
        print("max values", max_values)

        for max_value in max_values:
            row_index = max_value[0]
            column = max_value[1]
            column_values = df_accuracy_improvements[column].tolist()
            column_values[row_index] = "textbf{" + str(column_values[row_index]) + "}"

            df_accuracy_improvements[column] = np.array(column_values)

        print(df_accuracy_improvements)
        # df_table.columns = np.array(columns)
        print(df_accuracy_improvements.columns)

        indexes = ['CNN-a', 'CNN-b']
        for i in range(df_accuracy_improvements.shape[0]):
            row = df_accuracy_improvements.iloc[i]
            for index in indexes:
                value_string = row[index]
                add_textbf = False
                if "textbf{" in value_string:
                    value_string = value_string.replace("textbf{", "").replace("}", "")
                    add_textbf = True

                if ")" in value_string:
                    value_string = value_string.replace("(", "").split(")")
                    gain = value_string[0]
                    acc = value_string[1]
                else:
                    gain = ""
                    acc = value_string

                if add_textbf:
                    if gain != "":
                        gain = "textbf{" + gain + "}"
                    acc = "textbf{" + acc + "}"

                row[index] = acc + " & " + gain

            df_accuracy_improvements.iloc[i] = row

        latex = df_accuracy_improvements.to_latex().replace("\\\nEMNIST", "\\\n\hline\nEMNIST").replace("\\\nCIFAR-10", "\\\n\hline\nCIFAR-10").replace("\\bottomrule", "\\hline\n\\bottomrule").replace("\\midrule", "\\hline\n\\midrule").replace("\\toprule", "\\hline\n\\toprule").replace("textbf", r"\textbf").replace("\}", "}").replace("\{", "{").replace("\\begin{tabular", "\\resizebox{\columnwidth}{!}{\\begin{tabular}").replace("\$", "$").replace("textuparrow", "\oitextuparrow").replace("textdownarrow", "\oitextdownarrow").replace("\&", "&").replace("\_", "_")
        if 'Acc' in target_col:
            latex = latex.replace("\oitextuparrow0.0", "0.0")
        else:
            latex = latex.replace("\oitextuparrow0.0\%", "")

        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/csv/""".format(
            self.experiment, str(self.dataset), self.num_clients, self.num_rounds, self.fraction_fit,
            str(self.model_name),
            str(self.alpha), self.comment)
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        filename = """{}latex_{}_{}.txt""".format(base_dir, str(alpha), target_col)
        print("ddr: ", filename)
        pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

    def idmax(self, df, n_solutions, range_of_string, target_col):

        df_indexes = []
        columns = df.columns.tolist()
        print("colunas", columns)
        for i in range(len(columns)):
            column = columns[i]
            column_values = df[column].tolist()
            print("ddd", column_values)
            indexes = self.select_mean(i, column_values, columns, n_solutions, range_of_string, target_col)
            df_indexes += indexes

        return df_indexes

    def select_mean(self, index, column_values, columns, n_solutions, range_of_string, target_col):

        list_of_means = []
        indexes = []
        print("ola: ", column_values, "ola0")

        for i in range(len(column_values)):
            print("valor: ", column_values[i])
            value = float(str(column_values[i])[:range_of_string])
            interval = float(str(column_values[i])[range_of_string+1:range_of_string+4])
            minimum = value - interval
            maximum = value + interval
            list_of_means.append((value, minimum, maximum))

        for i in range(0, len(list_of_means), n_solutions):

            dataset_values = list_of_means[i: i + n_solutions]
            if 'Acc' in target_col:
                max_tuple = max(dataset_values, key=lambda e: e[0])
            else:
                max_tuple = min(dataset_values, key=lambda e: e[0])
            column_min_value = max_tuple[1]
            column_max_value = max_tuple[2]
            print("maximo: ", column_max_value)
            for j in range(len(list_of_means)):
                value_tuple = list_of_means[j]
                min_value = value_tuple[1]
                max_value = value_tuple[2]
                if j >= i and j < i + n_solutions:
                    if not (max_value < column_min_value or min_value > column_max_value):
                        indexes.append([j, columns[index]])

        return indexes

    def filter(self, df, model, dataset, alpha, shared_layer=None):

        # df['Accuracy (%)'] = df['Accuracy (%)']*100
        if strategy is not None:
            df = df.query(
                """Dataset=='{}' and Model=='{}'""".format(str(dataset), model))
            df = df[df['\u03B1'] == alpha]
            df = df[df['Solution'] == shared_layer]
        else:
            df = df.query(
                """Dataset=='{}' and Model=='{}'""".format(dataset), model)
            df = df[df['\u03B1'] == alpha]

        print("filtrou: ", df)

        return df


    def evaluate_client_joint_accuracy(self, df, alpha):

        # df = df[df['Solution'] == "$FedAvg+FP_d$"]

        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/""".format(
            self.experiment, str(self.dataset), self.num_clients, self.num_rounds, self.fraction_fit,
            str(self.model_name),
            str(self.alpha), self.comment)

        x_column = 'Round'
        y_column = 'Accuracy (%)'
        hue = 'Solution'
        style = None

        df = df.query("""\u03B1 == {}""".format(alpha))
        print("strategias unicas: ", df['Solution'].unique().tolist())

        if self.experiment == "dls_compredict":
            compression = ["$FedAvg+FP_{dc}$", "$FedAvg+FP_{d}$", "$FedAvg+FP_{c}$", "$FedAvg+FP$", "$FedAvg$", "$FedAvg+FP_{kd}$", "$FedAvg+FP_{s}$", "$FedAvg+FP_{per}$"]
        elif -1 in self.compression:
            compression = ["$FedAvg+FP_{d}$", 'FedPredict', 'FedAvg']
        else:
            compression = ["$FedAvg+FP_{c}$", 'FedPredict', 'FedAvg']

        # print(compression)
        # exit()

        if len(self.dataset) >= 2:

            print("testar1")
            print(df[df['Solution'] == "$FedAvg+FP_{d}$"])

            fig, ax = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(6, 6))
            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[0])
            x = df[x_column].tolist()
            y = df[y_column].tolist()
            print("jointplot: \n", df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[0])))
            line_plot(ax=ax[0, 0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[0])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=100,
                      y_min=0,
                      n=1)

            ax[0, 0].get_legend().remove()
            ax[0, 0].set_xlabel('')
            ax[0, 0].set_ylabel('')
            # ax[0, 0].set_xticks([])
            # ax[0, 0].set_yticks(np.arange(0, 101, 10))

            title = """{}; {}""".format(self.dataset_name_list[1], self.model_name_list[0])
            line_plot(ax=ax[0, 1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[1], self.model_name_list[0])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=100,
                      y_min=0,
                      n=1)

            ax[0, 1].get_legend().remove()
            ax[0, 1].set_xlabel('')
            ax[0, 1].set_ylabel('')
            fig.suptitle("", fontsize=16)
            fig.supxlabel(x_column, y=-0.02)
            fig.supylabel(y_column, x=-0.005)

            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[1])
            line_plot(ax=ax[1, 0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[1])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=100,
                      y_min=0,
                      n=1)

            ax[1, 0].get_legend().remove()
            # ax[1, 0].legend(fontsize=7)
            ax[1, 0].set_xlabel('')
            ax[1, 0].set_ylabel('')

            title = """{}; {}""".format(self.dataset_name_list[1], self.model_name_list[1])
            line_plot(ax=ax[1, 1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[1], self.model_name_list[1])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=100,
                      y_min=0,
                      n=1)

            ax[1, 1].get_legend().remove()
            ax[1, 1].set_xlabel('')
            ax[1, 1].set_ylabel('')
            fig.suptitle("", fontsize=9)
            fig.supxlabel(x_column, y=-0.02)
            fig.supylabel(y_column, x=-0.005)

            lines_labels = [ax[1, 0].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            print("linhas")
            print(lines)
            print(lines[0].get_color(), lines[0].get_ls())
            print("rotulos")
            print(labels)
            # # exit()
            colors = []
            markers = []
            for i in range(len(lines)):
                color = lines[i].get_color()
                colors.append(color)
                ls = lines[i].get_ls()
                if ls not in ["o"]:
                    ls = "o"
            markers = ["", "-", "--"]

            f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
            handles = [f("o", colors[i]) for i in range(len(colors))]
            fig.legend(handles, labels, loc='upper center', ncol=4, title="""\u03B1={}""".format(alpha), bbox_to_anchor=(0.5, 1.06), fontsize=9)

            figure = fig.get_figure()
            Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
            Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
            print("de intere: ", base_dir)
            filename = """accuracy_varying_shared_layers_{}_lineplot_joint_{}_alpha_{}""".format(self.experiment, str(self.dataset), str(alpha))
            figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
            figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)

        else:
            fig, ax = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(10, 6))
            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[0])
            x = df[x_column].tolist()
            y = df[y_column].tolist()
            print("jointplot: \n",
                  df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[0])))
            line_plot(ax=ax[0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[0])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=100,
                      y_min=0,
                      n=1)

            ax[0].get_legend().remove()
            ax[0].set_xlabel('')
            ax[0].set_ylabel('')

            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[1])
            line_plot(ax=ax[1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[1])),
                      base_dir=base_dir,
                      file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot_joint",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=100,
                      y_min=0,
                      n=1)

            ax[1].get_legend().remove()
            ax[1].set_xlabel('')
            ax[1].set_ylabel('')
            fig.suptitle("", fontsize=16)
            fig.supxlabel(x_column, y=-0.02)
            fig.supylabel(y_column, x=-0.005)

            plt.subplots_adjust(wspace=0.07, hspace=0.14)
            lines_labels = [ax[0].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines, labels, loc='upper center', ncol=4, title="""\u03B1={}""".format(alpha),
                       bbox_to_anchor=(0.5, 1.05))
            # plt.xticks(np.arange(min(x), max(x) + 1, max(x) // 5))
            figure = fig.get_figure()
            Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
            Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
            filename = """accuracy_varying_shared_layers_{}_lineplot_joint_{}_alpha_{}""".format(self.experiment, str(self.dataset), str(alpha))
            figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
            figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)

    def similarity(self):


        df = self.df_concat_similarity
        print("Df similaridade: \n", df)
        max_layer = df['Layer'].max() - 1
        max_layer_dataset = {dataset: df.query("""Model == '{}'""".format(dataset))['Layer'].max() - 1 for dataset in self.model_name}
        # df = df.query("""Layer == 0 or Layer == {}""".format(max_layer))
        def summary(ag, d_f):


            deno = ag['Similarity'].tolist()[0]
            dataset = ag['Dataset'].tolist()[0]
            alpha = ag['\u03B1'].tolist()[0]
            model = ag['Model'].tolist()[0]
            max_layer = int(ag.query("""Model == '{}'""".format(model))['Layer'].max()) - 1
            round = ag['Round'].tolist()[0]
            # print("pergunta: ", """Round <= {} and Model == '{}' and \u03B1 == {} and Dataset == '{}' and Layer == {}""".format(round, model, alpha, dataset, 0))
            similarities_0 = np.mean(d_f.query("""Round <= {} and Model == '{}' and \u03B1 == {} and Dataset == '{}' and Layer == {}""".format(round, model, alpha, dataset, 0))['Similarity'].to_numpy())
            similarities_last = np.mean(d_f.query("""Round <= {} and Model == '{}' and \u03B1 == {} and Dataset == '{}' and Layer == {}""".format(round, model, alpha, dataset, max_layer))['Similarity'].to_numpy())
            # print("resultado: ", similarities_0, similarities_last, " maximo: ", max_layer)
            if deno == 0:
                deno = 1
            # print("rodar:")
            # print(df)
            # print("olha: ", similarities_0, similarities_last)

            # df = df.query("""Layer == 0 or Layer == {}""".format(max_layer))
            # layer_0 = df.query("Layer == 0")
            # layer__last = df.query("""Layer == {}""".format(max_layer))
            # if len(layer_0) < 1 or len(layer__last) < 1:
            #     print("vazio:")
            #     print(df, " maximo: ", max_layer)
            #     return
            # else:
            if similarities_0 is not None and similarities_last is not None:
                dif = abs(similarities_0 - similarities_last)
            else:
                dif = 1

            return pd.DataFrame({'df': [dif]})

        print("simi: ", max_layer)
        # print(df.to_string())
        df = df.groupby(['Round', 'Dataset', '\u03B1', 'Model']).apply(lambda e: summary(ag=e, d_f=df)).reset_index()

        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/""".format(
            self.experiment, str(self.dataset), self.num_clients, self.num_rounds, self.fraction_fit,
            str(self.model_name),
            str(self.alpha), self.comment)
        print("agrupou: ", df)

        if len(self.dataset) == 2:
            os.makedirs(base_dir + "png/", exist_ok=True)
            os.makedirs(base_dir + "svg/", exist_ok=True)
            os.makedirs(base_dir + "csv/", exist_ok=True)
            print("dataset == 2")
            print(df.iloc[0])
            x_column = 'Round'
            y_column = 'df'
            hue = '\u03B1'
            order = sorted(self.alpha)
            sci = True
            filename = """df_similarity_{}""".format(str(self.dataset))

            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[0])

            fig, ax = plt.subplots(2, 2,  sharex='all', sharey='all', figsize=(6, 6))
            print("endereco: ", base_dir)
            print("filename: ", filename)
            x = df[x_column].tolist()
            y = df[y_column].tolist()
            # print("filtrado:")
            # print(df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[0])).to_string())
            line_plot(ax=ax[0, 0], base_dir=base_dir, file_name=filename, title=title, df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[0])),
                     x_column=x_column, y_column=y_column, y_lim=True, y_max=1,
                     y_min=0, hue=hue, hue_order=order, type=1)
            ax[0, 0].get_legend().remove()
            ax[0, 0].set_xlabel('')
            ax[0, 0].set_ylabel('')
            # ax[0, 0].set_xticks([])
            # ax[0, 0].set_yticks(np.arange(0, 0.6, 0.1))
            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[1])

            line_plot(ax=ax[0, 1], base_dir=base_dir, file_name=filename, title=title, df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[1], self.model_name_list[0])),
                      x_column=x_column, y_column=y_column, y_lim=True, y_max=1,
                      y_min=0, hue=hue, hue_order=order, type=1)
            ax[0, 1].get_legend().remove()
            ax[0, 1].set_xlabel('')
            ax[0, 1].set_ylabel('')

            title = """{}; {}""".format(self.dataset_name_list[1], self.model_name_list[0])
            line_plot(ax=ax[1, 0], base_dir=base_dir, file_name=filename, title=title,
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[1])),
                      x_column=x_column, y_column=y_column, y_lim=True, y_max=1,
                      y_min=0, hue=hue, hue_order=order, type=1)
            ax[1, 0].get_legend().remove()
            ax[1, 0].set_xlabel('')
            ax[1, 0].set_ylabel('')
            # ax[0, 0].set_xticks([])
            # ax[0, 0].set_yticks(np.arange(0, 0.6, 0.1))
            title = """{}; {}""".format(self.dataset_name_list[1], self.model_name_list[1])

            line_plot(ax=ax[1, 1], base_dir=base_dir, file_name=filename, title=title,
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[1], self.model_name_list[1])),
                      x_column=x_column, y_column=y_column, y_lim=True, y_max=1,
                      y_min=0, hue=hue, hue_order=order, type=1)
            ax[1, 1].get_legend().remove()
            ax[1, 1].set_xlabel('')
            ax[1, 1].set_ylabel('')

            fig.suptitle("", fontsize=16)
            fig.supxlabel(x_column, y=-0.02)
            fig.supylabel(y_column, x=-0.005)
            # ax[0, 1].set_yticks(np.arange(0, 0.6, 0.1))
            # plt.tight_layout(pad=0.5)
            # ax[0, 1].set_xticks(np.arange(min(x), max(x) + 1, max(x) // 10))
            # plt.subplots_adjust(wspace=0.07, hspace=0.14)
            # lines_labels = [ax[0, 0].get_legend_handles_labels()]
            # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            # fig.legend(lines, labels, title=hue, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.03))
            lines_labels = [ax[0, 0].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            print("linhas")
            print(lines)
            print(lines[0].get_color(), lines[0].get_ls())
            print("rotulos")
            print(labels)
            # # exit()
            colors = []
            markers = []
            for i in range(len(lines)):
                color = lines[i].get_color()
                colors.append(color)
                ls = lines[i].get_ls()
                if ls not in ["o"]:
                    ls = "o"
            markers = ["", "-", "--"]

            f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
            handles = [f("o", colors[i]) for i in range(2)]
            ax[0, 0].legend(handles, labels, fontsize=7, ncols=2, title='\u03B1')


            figure = fig.get_figure()
            Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
            Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
            figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
            figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)

        else:
            os.makedirs(base_dir + "png/", exist_ok=True)
            os.makedirs(base_dir + "svg/", exist_ok=True)
            os.makedirs(base_dir + "csv/", exist_ok=True)
            print("dataset == 1")
            print(df.iloc[0])
            x_column = 'Round'
            y_column = 'df'
            hue = '\u03B1'
            order = sorted(self.alpha)
            sci = True
            filename = """df_similarity_{}""".format(str(self.dataset))

            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[0])

            fig, ax = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(6, 6))
            print("endereco: ", base_dir)
            print("filename: ", filename)
            x = df[x_column].tolist()
            y = df[y_column].tolist()
            print("filtrado:")
            print(df.query(
                """Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[0])).to_string())
            line_plot(ax=ax[0], base_dir=base_dir, file_name=filename, title=title,
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[0])),
                      x_column=x_column, y_column=y_column, y_lim=True, y_max=1,
                      y_min=0, hue=hue, hue_order=order, type=1)
            ax[0].get_legend().remove()
            ax[0].set_xlabel('')
            ax[0].set_ylabel('')
            # ax[0, 0].set_xticks([])
            # ax[0, 0].set_yticks(np.arange(0, 0.6, 0.1))
            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[1])

            line_plot(ax=ax[1], base_dir=base_dir, file_name=filename, title=title,
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[1])),
                      x_column=x_column, y_column=y_column, y_lim=True, y_max=1,
                      y_min=0, hue=hue, hue_order=order, type=1)
            ax[1].get_legend().remove()
            ax[1].set_xlabel('')
            ax[1].set_ylabel('')

            fig.suptitle("", fontsize=16)
            fig.supxlabel(x_column, y=-0.02)
            fig.supylabel(y_column, x=-0.005)
            # ax[0, 1].set_yticks(np.arange(0, 0.6, 0.1))
            # plt.tight_layout(pad=0.5)
            # ax[0, 1].set_xticks(np.arange(min(x), max(x) + 1, max(x) // 10))
            plt.subplots_adjust(wspace=0.07, hspace=0.14)
            lines_labels = [ax[0].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines, labels, title=hue, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.06))

            figure = fig.get_figure()
            Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
            Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
            figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
            figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)

    def evaluate_client_analysis_differnt_models(self, df):

        def summary(df):

            parameters = df['Parameters reduction (%)'].mean()
            accuracy_reduction = df['Accuracy reduction (%)'].mean()
            acc = df['Accuracy (%)'].mean()

            return pd.DataFrame({'Parameters reduction (%)': [parameters], 'Accuracy reduction (%)': [accuracy_reduction], 'Accuracy (%)': [acc]})

        df = df.groupby(['Dataset', 'Model', '\u03B1', 'Strategy', 'Solution', 'Round']).apply(summary).reset_index()

        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/""".format(
            self.experiment, str(self.dataset), self.num_clients, self.num_rounds, self.fraction_fit,
            str(self.model_name),
            str(self.alpha), self.comment)
        dataset = self.dataset_name_list[0]
        os.makedirs(base_dir + "png/", exist_ok=True)
        os.makedirs(base_dir + "svg/", exist_ok=True)
        os.makedirs(base_dir + "csv/", exist_ok=True)
        print("sumario")
        print(df.iloc[0])
        x_column = 'Model'
        y_column = 'Accuracy reduction (%)'
        hue = '\u03B1'
        order = ['CNN_6', 'CNN_10']
        sci = True
        filename = """evaluate_client_acc_reduction_alpha_model_{}""".format(dataset)
        title = """Accuracy reduction; Dataset={}""".format(dataset)
        bar_plot(base_dir=base_dir, file_name=filename, title=title, df=df,
                 x_column=x_column, y_column=y_column, y_lim=True, y_max=50,
                 y_min=-1, hue=hue, x_order=order)
        filename = """evaluate_client_acc_reduction_alpha_model_{}""".format(dataset)
        title = """Accuracy reduction; Dataset={}""".format(dataset)
        violin_plot(base_dir=base_dir, file_name=filename, title=title, df=df,
                 x_column=x_column, y_column=y_column, y_lim=True, y_max=10,
                 y_min=-1, hue=hue, x_order=order)
        # filename = """evaluate_client_acc_alpha_model_lineplot"""
        # y_column = "Accuracy (%)"
        # x_column = "Round"
        # hue = "Solution"
        # style = "\u03B1"
        # line_plot(df=df,
        #           base_dir=base_dir,
        #           file_name=filename,
        #           x_column=x_column,
        #           y_column=y_column,
        #           title=title,
        #           hue=hue,
        #           style=style,
        #           hue_order=None,
        #           type=1,
        #           log_scale=False,
        #           y_lim=True,
        #           y_max=100,
        #           y_min=0)

        x_column = 'Model'
        y_column = 'Parameters reduction (%)'
        hue = '\u03B1'
        order = ['CNN_6', 'CNN_10']
        sci = True
        filename = """evaluate_client_parameters_reduction_alpha_model_{}""".format(dataset)
        title = """Parameters reduction (%); Dataset={}""".format(dataset)
        bar_plot(base_dir=base_dir, file_name=filename, title=title, df=df, x_column=x_column, y_column=y_column,
                 y_lim=True, y_max=100, y_min=0, hue=hue, x_order=order)
        filename = """evaluate_client_parameters_reduction_alpha_model_{}""".format(dataset)
        title = """Parameters reduction (%); Dataset={}""".format(dataset)
        violin_plot(base_dir=base_dir, file_name=filename, title=title, df=df, x_column=x_column, y_column=y_column,
                 y_lim=True, y_max=100, y_min=0, hue=hue, x_order=order)
        # filename = """evaluate_client_parameters_alpha_model_lineplot"""
        # x_column = "Round"
        # y_column = "Parameters reduction (%)"
        # hue = "Solution"
        # style = "\u03B1"
        # line_plot(df=df,
        #           base_dir=base_dir,
        #           file_name=filename,
        #           x_column=x_column,
        #           y_column=y_column,
        #           title=title,
        #           hue=hue,
        #           hue_order=compression_methods,
        #           type=1,
        #           log_scale=False,
        #           y_lim=True,
        #           y_max=100,
        #           y_min=0)


    def evaluate_client_analysis_shared_layers(self, model, dataset):
        # acc
        print("iniciais: ", self.df_concat.columns)
        df = self.df_concat
        df = df.query("""Model == '{}' and Dataset == '{}'""".format(model, dataset))
        # def strategy(df):
        #     parameters = float(df['Size of parameters'].mean())/1000000
        #     config = float(df['Size of config'].mean())/1000000
        #     acc = float(df['Accuracy'].mean())
        #     acc_gain_per_byte = acc/parameters
        #     total_size = parameters + config
        #
        #     return pd.DataFrame({'Size of parameters (MB)': [parameters], 'Communication cost (MB)': [total_size], 'Accuracy': [acc], 'Accuracy gain per MB': [acc_gain_per_byte]})
        # df = df[['Accuracy', 'Round', 'Size of parameters', 'Size of config', 'Strategy', 'Solution', '\u03B1', 'Dataset', 'Model']].groupby(by=['Strategy', 'Solution', 'Round', '\u03B1', 'Dataset', 'Model']).apply(lambda e: strategy(e)).reset_index()[['Accuracy', 'Size of parameters (MB)', 'Communication cost (MB)', 'Strategy', 'Solution', 'Round', 'Accuracy gain per MB', '\u03B1', 'Dataset', 'Model']]
        # print("Com alpha: ", alpha, "\n", df)
        print("rodou: ", model, dataset)
        # if model == 'CNN-a' and dataset == 'EMNIST':
        #     print(df.query("Solution=='$FedAvg+FP_{s}$' and Model=='CNN-a' and Dataset=='EMNIST'"))
        #     exit()
        x_column = 'Round'
        y_column = 'Accuracy (%)'
        hue = 'Solution'
        comment = self.comment
        if comment == '':
            comment = 'bottom up'
        elif comment == 'inverted':
            comment = 'top down'
        elif comment == 'individual':
            comment = 'individual layer'
        else:
            comment = 'set'

        # df['Solution'] = df['Solution'].astype(int)
        # sort = {i: "" for i in df['Solution'].sort_values().unique().tolist()}
        # shared_layers_list = df['Solution'].tolist()
        # print("Lista: ", shared_layers_list)
        # for i in range(len(shared_layers_list)):
        #     shared_layer = str(shared_layers_list[i])
        #     if "-1" in shared_layer:
        #         shared_layers_list[i] = "$FedAvg+FP_{d}$"
        #         sort[shared_layer] = shared_layers_list[i]
        #         continue
        #     if "-2" in shared_layer:
        #         shared_layers_list[i] = "$FedAvg+FP_{dc}$"
        #         sort[shared_layer] = shared_layers_list[i]
        #         continue
        #     if "-3" in shared_layer:
        #         shared_layers_list[i] = "$FedAvg+FP_{c}$"
        #         sort[shared_layer] = shared_layers_list[i]
        #         continue
        #     new_shared_layer = "{"
        #     for layer in shared_layer:
        #         if len(new_shared_layer) == 1:
        #             new_shared_layer += layer
        #         else:
        #             new_shared_layer += ", " + layer
        #
        #     new_shared_layer += "}"
        #     if shared_layer == "10":
        #         new_shared_layer = "$FedAvg+FP$"
        #     if shared_layer == "50":
        #         new_shared_layer = "50% of the layers"
        #
        #     shared_layers_list[i] = new_shared_layer
        #     sort[shared_layer] = shared_layers_list[i]
        #
        # df['Solution'] = np.array(shared_layers_list)
        # compression_methods = list(sort.values())
        # sort = []
        # for i in compression_methods:
        #     if len(i) > 0:
        #         sort.append(i)
        # compression_methods = sort
        # print("ord: ", compression_methods)
        compression  = ["$FedAvg+FP_{dc}$", "$FedAvg+FP_{d}$", "$FedAvg+FP_{c}$", "$FedAvg+FP$", "$FedAvg$", "$FedAvg+FP_{s}$"]
        style = '\u03B1'

        title = """Accuracy in {}; Model={}""".format(dataset, model)
        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/""".format(self.experiment, dataset, self.num_clients, self.num_rounds, self.fraction_fit, model, alpha, self.comment)
        os.makedirs(base_dir + "png/", exist_ok=True)
        os.makedirs(base_dir + "svg/", exist_ok=True)
        os.makedirs(base_dir + "csv/", exist_ok=True)
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_acc_round_varying_shared_layers_lineplot" + "_ " + dataset + "_" + "_alpha" + str(alpha) + "_model_" + model,
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  hue_order=compression,
                  style=style,
                  type=1,
                  y_lim=True,
                  y_min=0,
                  y_max=100)
        # print("Custo {1}", df[df['Solution']=='{1}'])
        title = """Communication cost in {}; Model={}""".format(dataset, model)
        x_column = 'Round'
        y_column = 'Size of parameters (MB)'
        hue = 'Solution'
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_size_of_parameters_round_varying_shared_layers_lineplot" + "_ " + dataset + "_" + "_alpha" + str(alpha) + "_model_" + model,
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  hue_order=compression,
                  style=style,
                  type=1,
                  y_lim=False,
                  y_max=4,
                  y_min=0)

        if comment == "bottom up":
            # df = df[df["Solution"] > 1]
            pass
        # x_column = 'Round'
        # y_column = 'Accuracy gain per MB'
        # hue = 'Solution'
        # line_plot(df=df,
        #           base_dir=base_dir,
        #           file_name="evaluate_client_accuracy_gain_per_MB_varying_shared_layers_lineplot" + "_ " + dataset + "_" + "_alpha" + str(alpha) + "_model_" + model,
        #           x_column=x_column,
        #           y_column=y_column,
        #           title=title,
        #           hue=hue,
        #           hue_order=compression,
        #           type=1,
        #           log_scale=True)

        filename = base_dir + "csv/comparison.csv"
        df.to_csv(filename, index=False)
        print("antes: ", df.columns)
        aux = copy.deepcopy(df)
        if dataset == 'EMNIST' and model == 'CNN-b':
            print("e1: ", df.query("Solution=='$FedAvg+FP$' and Model=='CNN-b' and Dataset=='EMNIST'")[
                'Size of parameters (MB)'])
        if dataset == 'EMNIST' and model == 'CNN-b':
            print("e2: ", df.query("Solution=='$FedAvg+FP$' and Model=='CNN-b' and Dataset=='EMNIST'")[
                'Size of parameters (MB)'])
            exit()
        df_preprocessed = copy.deepcopy(df)

        df = df[df['Solution'] != "$FedAvg+FP$"]
        df = df[df['Solution'] != "{1}"]
        # compression_methods =  ['FedPredict (with ALS)']
        compression = ["$FedAvg+FP_{dc}$"]
        print("menor: ", df['Accuracy reduction (%)'].min())
        print("Fed", df[df['Solution'] == 'FedPredict (with ALS)'][['Accuracy reduction (%)', 'Round']])
        print("tra: ", df['Solution'].unique().tolist())

        x_column = 'Round'
        y_column = 'Accuracy reduction (%)'
        hue = 'Solution'
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_accuracy_reduction_varying_shared_layers_lineplot" + "_ " + dataset + "_" + "_alpha" + str(alpha) + "_model_" + model,
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  hue_order=compression,
                  type=1,
                  log_scale=False,
                  y_lim=False,
                  y_max=10,
                  y_min=-3)

        x_column = 'Round'
        y_column = 'Parameters reduction (MB)'
        hue = 'Solution'
        style = '\u03B1'
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_Parameters_reduction_varying_shared_layers_lineplot" + "_ " + dataset + "_" + "_alpha" + str(alpha) + "_model_" + model,
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  style=style,
                  hue_order=compression,
                  type=1,
                  log_scale=True,
                  y_lim=False,
                  y_max=4,
                  y_min=0,
                  n=1)

        x_column = 'Round'
        y_column = 'Parameters reduction (%)'
        hue = 'Solution'
        style = '\u03B1'
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_Parameters_reduction_percentage_varying_shared_layers_lineplot" + "_ " + dataset + "_" + "_alpha" + str(
                      alpha) + "_model_" + model,
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  style=style,
                  hue_order=compression,
                  type=1,
                  log_scale=False,
                  y_lim=True,
                  y_max=100,
                  y_min=20,
                  n=1)

        return df_preprocessed


if __name__ == '__main__':
    """
        This code generates a joint plot (multiples plots in one image) and a table of accuracy.
        It is done for each experiment.
    """

    strategy = "FedPredict"
    type_model = "torch"
    aggregation_method = "None"
    fraction_fit = 0.3
    num_clients = 20
    model_name = ["CNN_2", "CNN_3"]
    dataset = ["EMNIST", "CIFAR10", "GTSRB"]
    alpha = [0.1, 1.0]
    num_rounds = 100
    epochs = 1
    # compression_methods = [-1, 1, 2, 3, 4, 12, 13, 14, 123, 124, 134, 23, 24, 1234, 34]
    #compression_methods = [1, 12, 123, 1234]
    # compression_methods = [4, 34, 234, 1234]
    compression = ["no", "sparsification", "fedkd", "per", "dls", "compredict", "dls_compredict"]
    # "fedkd", "sparsification", "per",
    comment = "set"

    Varying_Shared_layers(tp=type_model, strategy_name=strategy, fraction_fit=fraction_fit, aggregation_method=aggregation_method, new_clients=False, new_clients_train=False, num_clients=num_clients,
                          model_name=model_name, dataset=dataset, class_per_client=2, alpha=alpha, num_rounds=num_rounds, epochs=epochs,
                          comment=comment, compression=compression).start()
