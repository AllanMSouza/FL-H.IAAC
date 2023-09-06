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
                 class_per_client, alpha, num_rounds, epochs, comment, layer_selection_evaluate):

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
        self.layer_selection_evaluate = layer_selection_evaluate
        self.model_name_list = [model.replace("CNN_2", "CNN-a").replace("CNN_3", "CNN-b") for model in self.model_name]
        self.dataset_name_list = [dataset.replace("CIFAR10", "CIFAR-10") for dataset in self.dataset]
        if -1 in self.layer_selection_evaluate and -2 in self.layer_selection_evaluate:
            self.experiment = "als_compredict"
        elif -1 in self.layer_selection_evaluate:
            self.experiment = "als"
        elif -2 in self.layer_selection_evaluate:
            self.experiment = "compredict"

    def start(self):

        self.build_filenames()

        df_concat = None

        for model in self.model_name:
            for dataset in self.dataset:
                model_name = model.replace("CNN_2", "CNN-a").replace("CNN_3", "CNN-b")
                dataset_name = dataset.replace("CIFAR10", "CIFAR-10")
                df = self.evaluate_client_analysis_shared_layers(model_name, dataset_name)
                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df], ignore_index=True)

        print(df_concat)
        self.evaluate_client_analysis_differnt_models(df_concat)
        self.evaluate_client_joint_parameter_reduction(df_concat)
        alphas = df_concat['\u03B1'].unique().tolist()
        models = df_concat['Model'].unique().tolist()
        df_concat = self.build_filename_fedavg(df_concat)
        for alpha in alphas:
            self.evaluate_client_joint_accuracy(df_concat, alpha)
            self.joint_table(self.build_filename_fedavg(self.df_concat, use_mean=False), alpha=alpha, models=models)

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
            if "-1" in shared_layer:
                shared_layers_list[i] = "$FedPredict_{d}$"
                continue
            if "-2" in shared_layer:
                shared_layers_list[i] = "$FedPredict_{dc}$"
                continue
            if "-3" in shared_layer:
                shared_layers_list[i] = "$FedPredict_{c}$"
                continue
            new_shared_layer = "{"
            for layer in shared_layer:
                if len(new_shared_layer) == 1:
                    new_shared_layer += layer
                else:
                    new_shared_layer += ", " + layer

            new_shared_layer += "}"
            if shared_layer == "10":
                new_shared_layer = "$FedPredict$"
            if shared_layer == "50":
                new_shared_layer = "50% of the layers"

            shared_layers_list[i] = new_shared_layer

        df['Solution'] = np.array(shared_layers_list)

        return df

    def build_filenames(self):

        files = ["evaluate_client.csv", "similarity_between_layers.csv", "norm.csv"]
        df_concat = None
        df_concat_similarity = None
        df_concat_norm = None
        for layers in self.layer_selection_evaluate:
            for a in self.alpha:
                for model in self.model_name:
                    for dataset in self.dataset:
                        for file in files:
                            filename = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/FL-H.IAAC/" + f"logs/{self.type}/{self.strategy_name}-{self.aggregation_method}-{self.fraction_fit}/new_clients_{self.new_clients}_train_{self.new_clients_train}/{self.num_clients}/{model}/{dataset}/classes_per_client_{self.class_per_client}/alpha_{a}/{self.num_rounds}_rounds/{self.epochs}_local_epochs/{self.comment}_comment/{str(layers)}_layer_selection_evaluate/{file}"
                            df = pd.read_csv(filename)

                            # model_name = model.replace("CNN_2", "CNN-a").replace("CNN_3", "CNN-b")
                            # dataset_name = dataset.replace("CIFAR10", "CIFAR-10")
                            model_name = model.replace("CNN_2", "CNN-a").replace("CNN_3", "CNN-b")
                            dataset_name = dataset.replace("CIFAR10", "CIFAR-10")
                            print("olar: ", model_name, dataset_name, model, dataset)
                            if "evaluate" in file:
                                df['Solution'] = np.array([layers] * len(df))
                                df['Strategy'] = np.array([self.strategy_name] * len(df))
                                df['\u03B1'] = np.array([a]*len(df))
                                df['Model'] = np.array([model_name]*len(df))
                                df['Dataset'] = np.array([dataset_name]*len(df))
                                df['Accuracy (%)'] = df['Accuracy'].to_numpy() * 100
                                if df_concat is None:
                                    df_concat = df
                                else:
                                    df_concat = pd.concat([df_concat, df], ignore_index=True)
                            elif "similarity" in file:
                                if layers not in [-1, -2]:
                                    continue
                                df['\u03B1'] = np.array([a] * len(df))
                                df['Round'] = np.array(df['Server round'].tolist())
                                df['Dataset'] = np.array([dataset_name] * len(df))
                                df['Model'] = np.array([model_name] * len(df))
                                df['Solution'] = np.array([layers] * len(df))
                                df['Strategy'] = np.array([self.strategy_name] * len(df))
                                if df_concat_similarity is None:
                                    df_concat_similarity = df
                                else:
                                    df_concat_similarity = pd.concat([df_concat_similarity, df], ignore_index=True)
                            elif "norm" in file:
                                if layers not in [-1, -2]:
                                    continue
                                df['\u03B1'] = np.array([a] * len(df))
                                df['Server round'] = np.array(df['Round'].tolist())
                                df['Dataset'] = np.array([dataset_name] * len(df))
                                df['Model'] = np.array([model_name] * len(df))
                                df['Solution'] = np.array([layers] * len(df))
                                df['Strategy'] = np.array([self.strategy_name] * len(df))
                                if df_concat_norm is None:
                                    df_concat_norm = df
                                else:
                                    df_concat_norm = pd.concat([df_concat_norm, df], ignore_index=True)


        print("contruído: ", df_concat.columns)

        self.df_concat = self.convert_shared_layers(df_concat)
        self.df_concat_similarity = df_concat_similarity
        self.df_concat_norm = df_concat_norm
        # print("Leu similaridade", df_concat_similarity[['Round', '\u03B1', 'Similarity', 'Dataset', 'Model', 'Layer']].drop_duplicates().to_string())
        # exit()

    def build_filename_fedavg(self, df_concat, use_mean=True):

        files = ["evaluate_client.csv"]
        df_concat_similarity = None
        for layers in [min(self.layer_selection_evaluate)]:
            for a in self.alpha:
                for model in self.model_name:
                    for dataset in self.dataset:
                        for file in files:
                            filename = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/FL-H.IAAC/" + f"logs/{self.type}/FedAVG-{self.aggregation_method}-{self.fraction_fit}/new_clients_{self.new_clients}_train_{self.new_clients_train}/{self.num_clients}/{model}/{dataset}/classes_per_client_{self.class_per_client}/alpha_{a}/{self.num_rounds}_rounds/{self.epochs}_local_epochs/{self.comment}_comment/{str(-1)}_layer_selection_evaluate/{file}"
                            if not os.path.exists(filename):
                                print("não achou fedavg")
                                return df_concat
                            df = pd.read_csv(filename)
                            model_name = model.replace("CNN_2", "CNN-a").replace("CNN_3", "CNN-b")
                            dataset_name = dataset.replace("CIFAR10", "CIFAR-10")
                            if "evaluate" in file:
                                df['Solution'] = np.array([layers] * len(df))
                                df['Strategy'] = np.array(['FedAvg'] * len(df))
                                df['Solution'] = np.array(["$FedAvg$"] * len(df))
                                df['\u03B1'] = np.array([a]*len(df))
                                df['Model'] = np.array([model_name]*len(df))
                                df['Dataset'] = np.array([dataset_name]*len(df))
                                df['Accuracy (%)'] = df['Accuracy'].to_numpy() * 100

                                def summary(df):

                                    acc = df['Accuracy (%)'].mean()

                                    return pd.DataFrame({'Accuracy (%)': [acc]})

                                if use_mean:
                                    df = df.groupby(
                                        ['Dataset', 'Model', '\u03B1', 'Strategy', 'Solution', 'Round']).mean().reset_index()

                                if df_concat is None:
                                    df_concat = df
                                else:
                                    df_concat = pd.concat([df_concat, df], ignore_index=True)

        print("Concatenado: ", df_concat.to_string())

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

        if len(self.dataset) == 2 and len(self.model_name) == 2 and -2 in self.layer_selection_evaluate:
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

        if len(self.dataset) == 2 and len(self.model_name) == 2 and -2 in self.layer_selection_evaluate:
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

        # df = df[df['Solution'] == "$FedPredict_d$"]
        df = df[df['Solution'] != "$FedPredict$"]
        fig, ax = plt.subplots(2, 2,  sharex='all', sharey='all', figsize=(6, 6))

        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/""".format(
            self.experiment, str(self.dataset), self.num_clients, self.num_rounds, self.fraction_fit,
            str(self.model_name),
            str(self.alpha), self.comment)

        x_column = 'Round'
        y_column = 'Parameters reduction (%)'
        hue = 'Solution'
        style = '\u03B1'
        y_min = 0
        if self.experiment == "als_compredict":
            layer_selection_evaluate = ["$FedPredict_{dc}$", "$FedPredict_{d}$", "$FedPredict_{c}$"]
        elif -1 in self.layer_selection_evaluate:
            layer_selection_evaluate = ["$FedPredict_{d}$", 'FedPredict', 'FedAvg']
        else:
            layer_selection_evaluate = ["$FedPredict_{c}$", 'FedPredict', 'FedAvg']

        if "$FedPredict_{dc}$" not in df['Solution'].tolist():
            y_max = 60
        else:
            y_max = 100

        if len(self.dataset) >= 2:
            fig, ax = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(6, 6))
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
                      hue_order=layer_selection_evaluate,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=y_max,
                      y_min=y_min,
                      n=1)

            ax[0, 0].get_legend().remove()
            ax[0, 0].set_xlabel('')
            ax[0, 0].set_ylabel('')
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
                      hue_order=layer_selection_evaluate,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=y_max,
                      y_min=y_min,
                      n=1)

            ax[0, 1].get_legend().remove()
            ax[0, 1].set_xlabel('')
            ax[0, 1].set_ylabel('')

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
                      hue_order=layer_selection_evaluate,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=y_max,
                      y_min=y_min,
                      n=1)

            # ax[1, 0].get_legend().remove()
            ax[1, 0].legend(fontsize=7, ncol=2)
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
                      hue_order=layer_selection_evaluate,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=y_max,
                      y_min=y_min,
                      n=1)

            ax[1, 1].get_legend().remove()
            ax[1, 1].set_xlabel('')
            ax[1, 1].set_ylabel('')

            fig.suptitle("", fontsize=16)
            fig.supxlabel(x_column, y=-0.02)
            fig.supylabel(y_column, x=-0.005)
            # plt.tight_layout(pad=0.5)

            plt.subplots_adjust(wspace=0.07, hspace=0.14)
            lines_labels = [ax[0, 0].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            # fig.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.06))
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
                      # hue_order=layer_selection_evaluate,
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
                      # hue_order=layer_selection_evaluate,
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

        min_ = st.t.interval(alpha=ci, df=len(data) - 1,
                      loc=np.mean(data),
                      scale=st.sem(data))[0]

        mean = np.mean(data)
        average_variation = (mean - min_).round(1)
        mean = mean.round(1)

        return str(mean) + u"\u00B1" + str(average_variation)

    def joint_table(self, df, alpha, models):

        shared_layers = df['Solution'].unique().tolist()

        model_report = {i: {} for i in shared_layers}

        # df = df[df['Round'] == 100]
        print("receb: ", df.columns)
        df_test = df[
            ['Round', 'Size of parameters', 'Solution', 'Accuracy (%)', '\u03B1', 'Dataset', 'Model']]

        # df_test = df_test.query("""Round in [10, 100]""")
        print("agrupou table")
        print(df_test)
        convert_dict = {0.1: 5, 0.2: 10, 0.3: 15, 0.4: 20}
        # df_test['Fraction fit'] = np.array([convert_dict[i] for i in df_test['Fraction fit'].tolist()])

        columns = models

        index = [np.array(['EMNIST'] * len(columns) + ['CIFAR-10'] * len(columns)), np.array(columns * 2)]

        models_dict = {}
        ci = 0.95
        for shared_layer in model_report:

            mnist_acc = {}
            cifar10_acc = {}
            for column in columns:

                # mnist_acc[column] = (self.filter(df_test, experiment, 'MNIST', float(column), strategy=model_name)['Accuracy (%)']*100).mean().round(6)
                # cifar10_acc[column] = (self.filter(df_test, experiment, 'CIFAR10', float(column), strategy=model_name)['Accuracy (%)']*100).mean().round(6)
                mnist_acc[column] = self.t_distribution((self.filter(df_test, model=column, dataset='EMNIST', alpha=alpha, shared_layer=shared_layer)[
                                         'Accuracy (%)']).tolist(), ci)
                cifar10_acc[column] = self.t_distribution((self.filter(df_test,model=column, dataset='CIFAR-10', alpha=alpha, shared_layer=shared_layer)[
                                           'Accuracy (%)']).tolist(), ci)

            model_metrics = []

            for column in columns:
                model_metrics.append(mnist_acc[column])
            for column in columns:
                model_metrics.append(cifar10_acc[column])

            models_dict[shared_layer] = model_metrics

        df_table = pd.DataFrame(models_dict, index=index).round(4)
        print(df_table.to_string())



        max_values = self.idmax(df_table)
        print("max values", max_values)

        for max_value in max_values:
            row_index = max_value[0]
            column = max_value[1]
            column_values = df_table[column].tolist()
            column_values[row_index] = "textbf{" + str(column_values[row_index]) + "}"

            df_table[column] = np.array(column_values)

        print(df_table)
        df_table.columns = np.array(["$FedPredict_{dc}$", "$FedPredict_{d}$", "$FedPredict_{c}$", "$FedPredict$", "$FedAvg$"])
        print(df_table.columns)

        latex = df_table.to_latex().replace("\\\nMNIST", "\\\n\hline\nMNIST").replace("\\\nCIFAR-10", "\\\n\hline\nCIFAR-10").replace("\\bottomrule", "\\hline\n\\bottomrule").replace("\\midrule", "\\hline\n\\midrule").replace("\\toprule", "\\hline\n\\toprule").replace("textbf", r"\textbf").replace("\}", "}").replace("\{", "{").replace("\\begin{tabular", "\\resizebox{\columnwidth}{!}{\\begin{tabular}")

        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/csv/""".format(
            self.experiment, str(self.dataset), self.num_clients, self.num_rounds, self.fraction_fit,
            str(self.model_name),
            str(self.alpha), self.comment)
        filename = """{}latex_{}.txt""".format(base_dir, str(alpha))
        pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

    def idmax(self, df):

        df_indexes = []
        columns = df.columns.tolist()
        print("colunas", columns)
        for i in range(len(df)):

            row = df.iloc[i].tolist()
            print("ddd", row)
            indexes = self.select_mean(i, row, columns)
            df_indexes += indexes

        return df_indexes

    def select_mean(self, index, values, columns):

        list_of_means = []
        indexes = []
        print("ola: ", values, "ola0")

        for i in range(len(values)):

            print("valor: ", values[i])
            value = float(str(values[i])[:4])
            list_of_means.append(value)

        max_value = max(list_of_means)
        print("maximo: ", max_value)
        for i in range(len(list_of_means)):

            if list_of_means[i] == max_value:
                indexes.append([index, columns[i]])

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

        # df = df[df['Solution'] == "$FedPredict_d$"]

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

        if self.experiment == "als_compredict":
            layer_selection_evaluate = ["$FedPredict_{dc}$", "$FedPredict_{d}$", "$FedPredict_{c}$", "$FedPredict$", "$FedAvg$"]
        elif -1 in self.layer_selection_evaluate:
            layer_selection_evaluate = ["$FedPredict_{d}$", 'FedPredict', 'FedAvg']
        else:
            layer_selection_evaluate = ["$FedPredict_{c}$", 'FedPredict', 'FedAvg']

        if len(self.dataset) >= 2:

            print("testar1")
            print(df[df['Solution'] == "$FedPredict_{d}$"])

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
                      hue_order=layer_selection_evaluate,
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
                      hue_order=layer_selection_evaluate,
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
            # ax[0, 1].set_yticks([])
            # ax[0, 1].set_xticks([])
            # plt.tight_layout(pad=0.5)

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
                      hue_order=layer_selection_evaluate,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=100,
                      y_min=0,
                      n=1)

            # ax[1, 0].get_legend().remove()
            ax[1, 0].legend(fontsize=7)
            ax[1, 0].set_xlabel('')
            ax[1, 0].set_ylabel('')
            # ax[1, 0].set_xticks([])
            # ax[1, 0].set_yticks(np.arange(0, 101, 10))
            # ax[1, 0].set_xticks(np.arange(0, max(x) + 1, 5))

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
                      hue_order=layer_selection_evaluate,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=100,
                      y_min=0,
                      n=1)

            ax[1, 1].get_legend().remove()
            ax[1, 1].set_xlabel('')
            ax[1, 1].set_ylabel('')
            fig.suptitle("", fontsize=16)
            fig.supxlabel(x_column, y=-0.02)
            fig.supylabel(y_column, x=-0.005)
            # ax[1, 1].set_yticks([])
            # ax[1, 1].set_xticks(np.arange(0, max(x) + 1, 5))
            # ax[1].set_xticks([])
            # plt.tight_layout(pad=0.5)



            plt.subplots_adjust(wspace=0.07, hspace=0.14)
            lines_labels = [ax[0, 0].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            # fig.legend(lines, labels, loc='upper center', ncol=4, title="""\u03B1={}""".format(alpha), bbox_to_anchor=(0.5, 1.05))
            fig.suptitle("""\u03B1={}""".format(alpha))
            # plt.xticks(np.arange(min(x), max(x) + 1, max(x) // 5))
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
                      hue_order=layer_selection_evaluate,
                      type=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=100,
                      y_min=0,
                      n=1)

            ax[0].get_legend().remove()
            ax[0].set_xlabel('')
            ax[0].set_ylabel('')
            # ax[0, 0].set_xticks([])
            # ax[0, 0].set_yticks(np.arange(0, 101, 10))

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
                      hue_order=layer_selection_evaluate,
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
            print("filtrado:")
            print(df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[0])).to_string())
            line_plot(ax=ax[0, 0], base_dir=base_dir, file_name=filename, title=title, df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[0], self.model_name_list[0])),
                     x_column=x_column, y_column=y_column, y_lim=True, y_max=1,
                     y_min=0, hue=hue, hue_order=order, type=1)
            ax[0, 0].get_legend().remove()
            ax[0, 0].set_xlabel('')
            ax[0, 0].set_ylabel('')
            # ax[0, 0].set_xticks([])
            # ax[0, 0].set_yticks(np.arange(0, 0.6, 0.1))
            title = """{}; {}""".format(self.dataset_name_list[1], self.model_name_list[0])

            line_plot(ax=ax[0, 1], base_dir=base_dir, file_name=filename, title=title, df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset_name_list[1], self.model_name_list[0])),
                      x_column=x_column, y_column=y_column, y_lim=True, y_max=1,
                      y_min=0, hue=hue, hue_order=order, type=1)
            ax[0, 1].get_legend().remove()
            ax[0, 1].set_xlabel('')
            ax[0, 1].set_ylabel('')

            title = """{}; {}""".format(self.dataset_name_list[0], self.model_name_list[1])
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
            plt.subplots_adjust(wspace=0.07, hspace=0.14)
            lines_labels = [ax[0, 0].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines, labels, title=hue, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.03))


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
        #           hue_order=layer_selection_evaluate,
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
        def strategy(df):
            parameters = float(df['Size of parameters'].mean())/1000000
            config = float(df['Size of config'].mean())/1000000
            acc = float(df['Accuracy'].mean())
            acc_gain_per_byte = acc/parameters
            total_size = parameters + config

            return pd.DataFrame({'Size of parameters (MB)': [parameters], 'Communication cost (MB)': [total_size], 'Accuracy': [acc], 'Accuracy gain per MB': [acc_gain_per_byte]})
        df = df[['Accuracy', 'Round', 'Size of parameters', 'Size of config', 'Strategy', 'Solution', '\u03B1', 'Dataset', 'Model']].groupby(by=['Strategy', 'Solution', 'Round', '\u03B1', 'Dataset', 'Model']).apply(lambda e: strategy(e)).reset_index()[['Accuracy', 'Size of parameters (MB)', 'Communication cost (MB)', 'Strategy', 'Solution', 'Round', 'Accuracy gain per MB', '\u03B1', 'Dataset', 'Model']]
        # print("Com alpha: ", alpha, "\n", df)
        df['Accuracy (%)'] = df['Accuracy'] * 100
        df['Accuracy (%)'] = df['Accuracy (%)'].round(4)
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
        #         shared_layers_list[i] = "$FedPredict_{d}$"
        #         sort[shared_layer] = shared_layers_list[i]
        #         continue
        #     if "-2" in shared_layer:
        #         shared_layers_list[i] = "$FedPredict_{dc}$"
        #         sort[shared_layer] = shared_layers_list[i]
        #         continue
        #     if "-3" in shared_layer:
        #         shared_layers_list[i] = "$FedPredict_{c}$"
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
        #         new_shared_layer = "$FedPredict$"
        #     if shared_layer == "50":
        #         new_shared_layer = "50% of the layers"
        #
        #     shared_layers_list[i] = new_shared_layer
        #     sort[shared_layer] = shared_layers_list[i]
        #
        # df['Solution'] = np.array(shared_layers_list)
        # layer_selection_evaluate = list(sort.values())
        # sort = []
        # for i in layer_selection_evaluate:
        #     if len(i) > 0:
        #         sort.append(i)
        # layer_selection_evaluate = sort
        # print("ord: ", layer_selection_evaluate)
        layer_selection_evaluate  = ["$FedPredict_{dc}$", "$FedPredict_{d}$", "$FedPredict_{c}$", "$FedPredict$", "$FedAvg$"]
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
                  hue_order=layer_selection_evaluate,
                  style=style,
                  type=1,
                  y_lim=True,
                  y_min=0,
                  y_max=100)
        # print("Custo {1}", df[df['Solution']=='{1}'])
        title = """Communication cost in {}; Model={}""".format(dataset, model)
        x_column = 'Round'
        y_column = 'Communication cost (MB)'
        hue = 'Solution'
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_communication_cost_round_varying_shared_layers_lineplot" + "_ " + dataset + "_" + "_alpha" + str(alpha) + "_model_" + model,
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  hue_order=layer_selection_evaluate,
                  style=style,
                  type=1,
                  y_lim=False,
                  y_max=4,
                  y_min=0)

        if comment == "bottom up":
            # df = df[df["Solution"] > 1]
            pass
        print("Com alpha: ", alpha, "\n", df[['Accuracy', 'Solution', 'Round', 'Accuracy gain per MB']])
        x_column = 'Round'
        y_column = 'Accuracy gain per MB'
        hue = 'Solution'
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_accuracy_gain_per_MB_varying_shared_layers_lineplot" + "_ " + dataset + "_" + "_alpha" + str(alpha) + "_model_" + model,
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  hue_order=layer_selection_evaluate,
                  type=1,
                  log_scale=True)

        filename = base_dir + "csv/comparison.csv"
        df.to_csv(filename, index=False)

        def comparison_with_shared_layers(df, df_aux):

            round = int(df['Round'].values[0])
            dataset = str(df['Dataset'].values[0])
            alpha = float(df['\u03B1'].values[0])
            model = str(df['Model'].values[0])
            # print("interes: ", round, dataset, alpha, model)
            df_copy = copy.deepcopy(df_aux.query("""Round == {} and Dataset == '{}' and \u03B1 == {} and Model == '{}'""".format(round, dataset, alpha, model)))
            # print("apos: ", df_copy.columns)
            target = df_copy[df_copy['Solution'] == "$FedPredict$"]
            target_acc = target['Accuracy (%)'].tolist()[0]
            target_size = target['Size of parameters (MB)'].tolist()[0]
            acc = df['Accuracy (%)'].tolist()[0]
            accuracy = df['Accuracy (%)'].mean()
            size = df['Size of parameters (MB)'].tolist()[0]
            acc_reduction = target_acc - acc
            size_reduction = (target_size - size)
            size_reduction_percentage = (1 - size/target_size) * 100
            # acc_weight = 1
            # size_weight = 1
            # acc_score = acc_score *acc_weight
            # size_reduction = size_reduction * size_weight
            # score = 2*(acc_score * size_reduction)/(acc_score + size_reduction)
            # if df['Solution'].tolist()[0] == "{1, 2, 3, 4}":
            #     acc_reduction = 0.0001
            #     size_reduction = 0.0001

            return pd.DataFrame({'Accuracy reduction (%)': [acc_reduction], 'Parameters reduction (MB)': [size_reduction],
                                 'Parameters reduction (%)': [size_reduction_percentage], 'Accuracy (%)': [accuracy]})

        print("antes: ", df.columns)
        aux = copy.deepcopy(df)
        df = df[['Accuracy (%)', 'Size of parameters (MB)', 'Communication cost (MB)', 'Strategy', 'Solution', 'Round', 'Accuracy gain per MB', '\u03B1', 'Dataset', 'Model']].groupby(
            by=['Strategy', 'Round', 'Solution', 'Dataset', '\u03B1', 'Model']).apply(lambda e: comparison_with_shared_layers(df=e, df_aux=aux)).reset_index()[['Strategy', 'Round', 'Solution', '\u03B1', 'Accuracy (%)', 'Accuracy reduction (%)', 'Parameters reduction (MB)', 'Parameters reduction (%)', 'Dataset', 'Model']]

        df_preprocessed = copy.deepcopy(df)

        df = df[df['Solution'] != "$FedPredict$"]
        df = df[df['Solution'] != "{1}"]
        # layer_selection_evaluate =  ['FedPredict (with ALS)']
        layer_selection_evaluate = ["$FedPredict_{dc}$"]
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
                  hue_order=layer_selection_evaluate,
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
                  hue_order=layer_selection_evaluate,
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
                  hue_order=layer_selection_evaluate,
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
    dataset = ["EMNIST", "CIFAR10"]
    alpha = [0.1, 5.0]
    num_rounds = 100
    epochs = 1
    # layer_selection_evaluate = [-1, 1, 2, 3, 4, 12, 13, 14, 123, 124, 134, 23, 24, 1234, 34]
    #layer_selection_evaluate = [1, 12, 123, 1234]
    # layer_selection_evaluate = [4, 34, 234, 1234]
    layer_selection_evaluate = [-1, -2, -3, 10]
    comment = "set"

    Varying_Shared_layers(tp=type_model, strategy_name=strategy, fraction_fit=fraction_fit, aggregation_method=aggregation_method, new_clients=False, new_clients_train=False, num_clients=num_clients,
                          model_name=model_name, dataset=dataset, class_per_client=2, alpha=alpha, num_rounds=num_rounds, epochs=epochs,
                          comment=comment, layer_selection_evaluate=layer_selection_evaluate).start()
