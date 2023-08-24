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
        if -1 in self.layer_selection_evaluate:
            self.experiment = "als"
        elif -2 in self.layer_selection_evaluate:
            self.experiment = "compredict"

    def start(self):

        self.build_filenames()

        df_concat = None

        for model in self.model_name:
            for dataset in self.dataset:
                df = self.evaluate_client_analysis_shared_layers(model, dataset)
                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df], ignore_index=True)

        print(df_concat)
        self.evaluate_client_analysis_differnt_models(df_concat)
        self.evaluate_client_joint_parameter_reduction(df_concat)
        alphas = df_concat['Alpha'].unique().tolist()
        for alpha in alphas:
            self.evaluate_client_joint_accuracy(self.build_filename_fedavg(df_concat), alpha)
            pass
        # for alpha in alphas:
        #     self.evaluate_client_joint_accuracy(df_concat, alpha)
        self.similarity()
        for alpha in alphas:
            self.evaluate_client_norm_analysis_nt(alpha)
        self.evaluate_client_norm_analysis()

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
                            if "evaluate" in file:
                                df['Shared layers'] = np.array([layers] * len(df))
                                df['Strategy'] = np.array([self.strategy_name] * len(df))
                                df['Alpha'] = np.array([a]*len(df))
                                df['Model'] = np.array([model]*len(df))
                                df['Dataset'] = np.array([dataset]*len(df))
                                if df_concat is None:
                                    df_concat = df
                                else:
                                    df_concat = pd.concat([df_concat, df], ignore_index=True)
                            elif "similarity" in file:
                                if layers not in [-1, -2]:
                                    continue
                                df['Alpha'] = np.array([a] * len(df))
                                df['Round'] = np.array(df['Server round'].tolist())
                                df['Dataset'] = np.array([dataset] * len(df))
                                df['Model'] = np.array([model] * len(df))
                                df['Shared layers'] = np.array([layers] * len(df))
                                df['Strategy'] = np.array([self.strategy_name] * len(df))
                                if df_concat_similarity is None:
                                    df_concat_similarity = df
                                else:
                                    df_concat_similarity = pd.concat([df_concat_similarity, df], ignore_index=True)
                            elif "norm" in file:
                                if layers not in [-1, -2]:
                                    continue
                                df['Alpha'] = np.array([a] * len(df))
                                df['Server round'] = np.array(df['Round'].tolist())
                                df['Dataset'] = np.array([dataset] * len(df))
                                df['Model'] = np.array([model] * len(df))
                                df['Shared layers'] = np.array([layers] * len(df))
                                df['Strategy'] = np.array([self.strategy_name] * len(df))
                                if df_concat_norm is None:
                                    df_concat_norm = df
                                else:
                                    df_concat_norm = pd.concat([df_concat_norm, df], ignore_index=True)


        print("contruído: ", df_concat.columns)

        self.df_concat = df_concat
        self.df_concat_similarity = df_concat_similarity
        self.df_concat_norm = df_concat_norm
        # print("Leu similaridade", df_concat_similarity[['Round', 'Alpha', 'Similarity', 'Dataset', 'Model', 'Layer']].drop_duplicates().to_string())
        # exit()

    def build_filename_fedavg(self, df_concat):

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
                            if "evaluate" in file:
                                df['Shared layers'] = np.array([layers] * len(df))
                                df['Strategy'] = np.array(['FedAvg'] * len(df))
                                df['Shared layers'] = np.array(['FedAvg'] * len(df))
                                df['Alpha'] = np.array([a]*len(df))
                                df['Model'] = np.array([model]*len(df))
                                df['Dataset'] = np.array([dataset]*len(df))
                                df['Accuracy (%)'] = df['Accuracy'].to_numpy() * 100

                                def summary(df):

                                    acc = df['Accuracy (%)'].mean()

                                    return pd.DataFrame({'Accuracy (%)': [acc]})

                                df = df.groupby(
                                    ['Dataset', 'Model', 'Alpha', 'Strategy', 'Shared layers', 'Round']).mean().reset_index()

                                if df_concat is None:
                                    df_concat = df
                                else:
                                    df_concat = pd.concat([df_concat, df], ignore_index=True)

        print("Concatenado: ", df_concat.to_string())

        return df_concat

    def evaluate_client_norm_analysis_nt(self, alpha):

        df = self.df_concat_norm.query("""Alpha == {}""".format(alpha))
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
            title = """{}; {}""".format(self.model_name[model_name_index], self.dataset[dataset_name_index])
            line_plot(
                        ax=ax[model_name_index, dataset_name_index],
                        df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[dataset_name_index], self.model_name[model_name_index])),
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
            title = """{}; {}""".format(self.model_name[model_name_index], self.dataset[dataset_name_index])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[dataset_name_index], self.model_name[model_name_index])),
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
            title = """{}; {}""".format(self.model_name[model_name_index], self.dataset[dataset_name_index])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[dataset_name_index], self.model_name[model_name_index])),
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
            title = """{}; {}""".format(self.model_name[model_name_index], self.dataset[dataset_name_index])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[dataset_name_index], self.model_name[model_name_index])),
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
            fig.legend(lines, labels, title="""nt""".format(alpha), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.04))
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
            hue = 'Alpha'
            hue_order = df[hue].unique().tolist().sort(reverse=False)

            model_name_index = 0
            dataset_name_index = 0
            title = """{}; {}""".format(self.model_name[model_name_index], self.dataset[dataset_name_index])
            print("agrupou")
            print("colunas: ", df.columns)
            print(df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[dataset_name_index],
                                                                           self.model_name[model_name_index])).groupby(["Round", "Alpha"]).mean().reset_index()[[x_column, y_column, hue]])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[dataset_name_index],
                                                                           self.model_name[model_name_index])).groupby(["Round", "Alpha"]).mean().reset_index()[[x_column, y_column, hue]],
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
            title = """{}; {}""".format(self.model_name[model_name_index], self.dataset[dataset_name_index])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[dataset_name_index],
                                                                           self.model_name[model_name_index])).groupby(
                    ["Round", "Alpha"]).mean().reset_index()[[x_column, y_column, hue]],
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
            title = """{}; {}""".format(self.model_name[model_name_index], self.dataset[dataset_name_index])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[dataset_name_index],
                                                                           self.model_name[model_name_index])).groupby(
                    ["Round", "Alpha"]).mean().reset_index()[[x_column, y_column, hue]],
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
            title = """{}; {}""".format(self.model_name[model_name_index], self.dataset[dataset_name_index])
            line_plot(
                ax=ax[model_name_index, dataset_name_index],
                df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[dataset_name_index],
                                                                           self.model_name[model_name_index])).groupby(
                    ["Round", "Alpha"]).mean().reset_index()[[x_column, y_column, hue]],
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
            fig.legend(lines, labels, title="""Alpha""".format(), loc='upper center', ncol=4,
                       bbox_to_anchor=(0.5, 1.06))
            figure = fig.get_figure()
            Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
            Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
            filename = """norm_round_{}_{}_alpha_{}""".format(
                self.experiment, str(self.dataset), alpha)
            figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
            figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)


    def evaluate_client_joint_parameter_reduction(self, df):

        # df = df[df['Shared layers'] == "FedPredict (with ALS)"]
        df = df[df['Shared layers'] != '100% of the layers']
        fig, ax = plt.subplots(2, 2,  sharex='all', sharey='all', figsize=(6, 6))

        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/""".format(
            self.experiment, str(self.dataset), self.num_clients, self.num_rounds, self.fraction_fit,
            str(self.model_name),
            str(self.alpha), self.comment)

        x_column = 'Round'
        y_column = 'Parameters reduction (%)'
        hue = 'Shared layers'
        style = 'Alpha'

        if len(self.dataset) >= 2:
            fig, ax = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(6, 6))
            title = """{}; {}""".format(self.dataset[0], self.model_name[0])
            x = df[x_column].tolist()
            y = df[y_column].tolist()
            print("jointplot: \n", df.query("""Dataset == '{}'""".format(self.dataset[0])))
            line_plot(ax=ax[0, 0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[0])),
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

            ax[0, 0].get_legend().remove()
            ax[0, 0].set_xlabel('')
            ax[0, 0].set_ylabel('')
            title = """{}; {}""".format(self.dataset[1], self.model_name[0])
            line_plot(ax=ax[0, 1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[1], self.model_name[0])),
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

            ax[0, 1].get_legend().remove()
            ax[0, 1].set_xlabel('')
            ax[0, 1].set_ylabel('')

            title = """{}; {}""".format(self.dataset[0], self.model_name[1])
            line_plot(ax=ax[1, 0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[1])),
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

            ax[1, 0].get_legend().remove()
            ax[1, 0].set_xlabel('')
            ax[1, 0].set_ylabel('')

            title = """{}; {}""".format(self.dataset[1], self.model_name[1])
            line_plot(ax=ax[1, 1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[1], self.model_name[1])),
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
            fig.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.06))
            figure = fig.get_figure()
            Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
            Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
            filename = """parameters_reduction_percentage_{}_varying_shared_layers_lineplot_joint_{}""".format(self.experiment, str(self.dataset))
            figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
            figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)

        else:
            fig, ax = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(8, 6))
            title = """{}; {}""".format(self.dataset[0], self.model_name[0])
            x = df[x_column].tolist()
            y = df[y_column].tolist()
            print("jointplot: \n", df.query("""Dataset == '{}'""".format(self.dataset[0])))
            line_plot(ax=ax[0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[0])),
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
            title = """{}; {}""".format(self.dataset[0], self.model_name[1])
            line_plot(ax=ax[1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[1])),
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
            fig.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.06))
            figure = fig.get_figure()
            Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
            Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
            filename = """parameters_reduction_percentage_{}_varying_shared_layers_lineplot_joint_{}""".format(self.experiment, str(self.dataset))
            figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
            figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)

    def evaluate_client_joint_accuracy(self, df, alpha):

        # df = df[df['Shared layers'] == "FedPredict (with ALS)"]

        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/""".format(
            self.experiment, str(self.dataset), self.num_clients, self.num_rounds, self.fraction_fit,
            str(self.model_name),
            str(self.alpha), self.comment)

        x_column = 'Round'
        y_column = 'Accuracy (%)'
        hue = 'Shared layers'
        style = None

        df = df.query("""Alpha == {}""".format(alpha))

        if -1 in self.layer_selection_evaluate:
            layer_selection_evaluate = ['FedPredict (with ALS)', '100% of the layers', 'FedAvg']
        else:
            layer_selection_evaluate = ['FedPredict (with ALS + Compredict)', '100% of the layers', 'FedAvg']

        if len(self.dataset) >= 2:
            fig, ax = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(6, 6))
            title = """{}; {}""".format(self.dataset[0], self.model_name[0])
            x = df[x_column].tolist()
            y = df[y_column].tolist()
            print("jointplot: \n", df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[0])))
            line_plot(ax=ax[0, 0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[0])),
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

            title = """{}; {}""".format(self.dataset[1], self.model_name[0])
            line_plot(ax=ax[0, 1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[1], self.model_name[0])),
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

            title = """{}; {}""".format(self.dataset[0], self.model_name[1])
            line_plot(ax=ax[1, 0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[1])),
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

            ax[1, 0].get_legend().remove()
            ax[1, 0].set_xlabel('')
            ax[1, 0].set_ylabel('')
            # ax[1, 0].set_xticks([])
            # ax[1, 0].set_yticks(np.arange(0, 101, 10))
            # ax[1, 0].set_xticks(np.arange(0, max(x) + 1, 5))

            title = """{}; {}""".format(self.dataset[1], self.model_name[1])
            line_plot(ax=ax[1, 1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[1], self.model_name[1])),
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
            fig.legend(lines, labels, loc='upper center', ncol=4, title="""Alpha={}""".format(alpha), bbox_to_anchor=(0.5, 1.05))
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
            title = """{}; {}""".format(self.dataset[0], self.model_name[0])
            x = df[x_column].tolist()
            y = df[y_column].tolist()
            print("jointplot: \n",
                  df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[0])))
            line_plot(ax=ax[0],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[0])),
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

            title = """{}; {}""".format(self.dataset[0], self.model_name[1])
            line_plot(ax=ax[1],
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[1])),
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
            fig.legend(lines, labels, loc='upper center', ncol=4, title="""Alpha={}""".format(alpha),
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
            alpha = ag['Alpha'].tolist()[0]
            model = ag['Model'].tolist()[0]
            max_layer = int(ag.query("""Model == '{}'""".format(model))['Layer'].max()) - 1
            round = ag['Round'].tolist()[0]
            print("pergunta: ", """Round <= {} and Model == '{}' and Alpha == {} and Dataset == '{}' and Layer == {}""".format(round, model, alpha, dataset, 0))
            similarities_0 = np.mean(d_f.query("""Round <= {} and Model == '{}' and Alpha == {} and Dataset == '{}' and Layer == {}""".format(round, model, alpha, dataset, 0))['Similarity'].to_numpy())
            similarities_last = np.mean(d_f.query("""Round <= {} and Model == '{}' and Alpha == {} and Dataset == '{}' and Layer == {}""".format(round, model, alpha, dataset, max_layer))['Similarity'].to_numpy())
            print("resultado: ", similarities_0, similarities_last, " maximo: ", max_layer)
            if deno == 0:
                deno = 1
            # print("rodar:")
            # print(df)
            print("olha: ", similarities_0, similarities_last)

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
        print(df.to_string())
        df = df.groupby(['Round', 'Dataset', 'Alpha', 'Model']).apply(lambda e: summary(ag=e, d_f=df)).reset_index()

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
            hue = 'Alpha'
            order = sorted(self.alpha)
            sci = True
            filename = """df_similarity_{}""".format(str(self.dataset))

            title = """{}; {}""".format(self.dataset[0], self.model_name[0])

            fig, ax = plt.subplots(2, 2,  sharex='all', sharey='all', figsize=(6, 6))
            print("endereco: ", base_dir)
            print("filename: ", filename)
            x = df[x_column].tolist()
            y = df[y_column].tolist()
            print("filtrado:")
            print(df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[0])).to_string())
            line_plot(ax=ax[0, 0], base_dir=base_dir, file_name=filename, title=title, df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[0])),
                     x_column=x_column, y_column=y_column, y_lim=True, y_max=1,
                     y_min=0, hue=hue, hue_order=order, type=1)
            ax[0, 0].get_legend().remove()
            ax[0, 0].set_xlabel('')
            ax[0, 0].set_ylabel('')
            # ax[0, 0].set_xticks([])
            # ax[0, 0].set_yticks(np.arange(0, 0.6, 0.1))
            title = """{}; {}""".format(self.dataset[1], self.model_name[0])

            line_plot(ax=ax[0, 1], base_dir=base_dir, file_name=filename, title=title, df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[1], self.model_name[0])),
                      x_column=x_column, y_column=y_column, y_lim=True, y_max=1,
                      y_min=0, hue=hue, hue_order=order, type=1)
            ax[0, 1].get_legend().remove()
            ax[0, 1].set_xlabel('')
            ax[0, 1].set_ylabel('')

            title = """{}; {}""".format(self.dataset[0], self.model_name[1])
            line_plot(ax=ax[1, 0], base_dir=base_dir, file_name=filename, title=title,
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[1])),
                      x_column=x_column, y_column=y_column, y_lim=True, y_max=1,
                      y_min=0, hue=hue, hue_order=order, type=1)
            ax[1, 0].get_legend().remove()
            ax[1, 0].set_xlabel('')
            ax[1, 0].set_ylabel('')
            # ax[0, 0].set_xticks([])
            # ax[0, 0].set_yticks(np.arange(0, 0.6, 0.1))
            title = """{}; {}""".format(self.dataset[1], self.model_name[1])

            line_plot(ax=ax[1, 1], base_dir=base_dir, file_name=filename, title=title,
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[1], self.model_name[1])),
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
            hue = 'Alpha'
            order = sorted(self.alpha)
            sci = True
            filename = """df_similarity_{}""".format(str(self.dataset))

            title = """{}; {}""".format(self.dataset[0], self.model_name[0])

            fig, ax = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(6, 6))
            print("endereco: ", base_dir)
            print("filename: ", filename)
            x = df[x_column].tolist()
            y = df[y_column].tolist()
            print("filtrado:")
            print(df.query(
                """Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[0])).to_string())
            line_plot(ax=ax[0], base_dir=base_dir, file_name=filename, title=title,
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[0])),
                      x_column=x_column, y_column=y_column, y_lim=True, y_max=1,
                      y_min=0, hue=hue, hue_order=order, type=1)
            ax[0].get_legend().remove()
            ax[0].set_xlabel('')
            ax[0].set_ylabel('')
            # ax[0, 0].set_xticks([])
            # ax[0, 0].set_yticks(np.arange(0, 0.6, 0.1))
            title = """{}; {}""".format(self.dataset[0], self.model_name[1])

            line_plot(ax=ax[1], base_dir=base_dir, file_name=filename, title=title,
                      df=df.query("""Dataset == '{}' and Model == '{}'""".format(self.dataset[0], self.model_name[1])),
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

        df = df.groupby(['Dataset', 'Model', 'Alpha', 'Strategy', 'Shared layers', 'Round']).apply(summary).reset_index()

        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}/{}_clients/{}_rounds/{}_fraction_fit/model_{}/alpha_{}/{}_comment/""".format(
            self.experiment, str(self.dataset), self.num_clients, self.num_rounds, self.fraction_fit,
            str(self.model_name),
            str(self.alpha), self.comment)
        dataset = self.dataset[0]
        os.makedirs(base_dir + "png/", exist_ok=True)
        os.makedirs(base_dir + "svg/", exist_ok=True)
        os.makedirs(base_dir + "csv/", exist_ok=True)
        print("sumario")
        print(df.iloc[0])
        x_column = 'Model'
        y_column = 'Accuracy reduction (%)'
        hue = 'Alpha'
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
        # hue = "Shared layers"
        # style = "Alpha"
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
        hue = 'Alpha'
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
        # hue = "Shared layers"
        # style = "Alpha"
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
        df = df[['Accuracy', 'Round', 'Size of parameters', 'Size of config', 'Strategy', 'Shared layers', 'Alpha', 'Dataset', 'Model']].groupby(by=['Strategy', 'Shared layers', 'Round', 'Alpha', 'Dataset', 'Model']).apply(lambda e: strategy(e)).reset_index()[['Accuracy', 'Size of parameters (MB)', 'Communication cost (MB)', 'Strategy', 'Shared layers', 'Round', 'Accuracy gain per MB', 'Alpha', 'Dataset', 'Model']]
        # print("Com alpha: ", alpha, "\n", df)
        df['Accuracy (%)'] = df['Accuracy'] * 100
        df['Accuracy (%)'] = df['Accuracy (%)'].round(4)
        x_column = 'Round'
        y_column = 'Accuracy (%)'
        hue = 'Shared layers'
        comment = self.comment
        if comment == '':
            comment = 'bottom up'
        elif comment == 'inverted':
            comment = 'top down'
        elif comment == 'individual':
            comment = 'individual layer'
        else:
            comment = 'set'

        df['Shared layers'] = df['Shared layers'].astype(int)
        sort = {i: "" for i in df['Shared layers'].sort_values().unique().tolist()}
        shared_layers_list = df['Shared layers'].tolist()
        print("Lista: ", shared_layers_list)
        for i in range(len(shared_layers_list)):
            shared_layer = str(shared_layers_list[i])
            if "-1" in shared_layer:
                shared_layers_list[i] = "FedPredict (with ALS)"
                sort[shared_layer] = shared_layers_list[i]
                continue
            if "-2" in shared_layer:
                shared_layers_list[i] = "FedPredict (with ALS + Compredict)"
                sort[shared_layer] = shared_layers_list[i]
                continue
            new_shared_layer = "{"
            for layer in shared_layer:
                if len(new_shared_layer) == 1:
                    new_shared_layer += layer
                else:
                    new_shared_layer += ", " + layer

            new_shared_layer += "}"
            if shared_layer == "10":
                new_shared_layer = "100% of the layers"
            if shared_layer == "50":
                new_shared_layer = "50% of the layers"

            shared_layers_list[i] = new_shared_layer
            sort[shared_layer] = shared_layers_list[i]

        df['Shared layers'] = np.array(shared_layers_list)
        layer_selection_evaluate = list(sort.values())
        sort = []
        for i in layer_selection_evaluate:
            if len(i) > 0:
                sort.append(i)
        layer_selection_evaluate = sort
        print("ord: ", layer_selection_evaluate)
        style = 'Alpha'

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
        # print("Custo {1}", df[df['Shared layers']=='{1}'])
        title = """Communication cost in {}; Model={}""".format(dataset, model)
        x_column = 'Round'
        y_column = 'Communication cost (MB)'
        hue = 'Shared layers'
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
            # df = df[df["Shared layers"] > 1]
            pass
        print("Com alpha: ", alpha, "\n", df[['Accuracy', 'Shared layers', 'Round', 'Accuracy gain per MB']])
        x_column = 'Round'
        y_column = 'Accuracy gain per MB'
        hue = 'Shared layers'
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
            print("fr: ", df.columns)
            round = int(df['Round'].values[0])
            dataset = str(df['Dataset'].values[0])
            alpha = float(df['Alpha'].values[0])
            model = str(df['Model'].values[0])

            df_copy = copy.deepcopy(df_aux.query("""Round == {} and Dataset == '{}' and Alpha == {} and Model == '{}'""".format(round, dataset, alpha, model)))
            print("apos: ", df_copy.columns)
            target = df_copy[df_copy['Shared layers'] == "100% of the layers"]
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
            # if df['Shared layers'].tolist()[0] == "{1, 2, 3, 4}":
            #     acc_reduction = 0.0001
            #     size_reduction = 0.0001

            return pd.DataFrame({'Accuracy reduction (%)': [acc_reduction], 'Parameters reduction (MB)': [size_reduction],
                                 'Parameters reduction (%)': [size_reduction_percentage], 'Accuracy (%)': [accuracy]})

        print("antes: ", df.columns)
        aux = copy.deepcopy(df)
        df = df[['Accuracy (%)', 'Size of parameters (MB)', 'Communication cost (MB)', 'Strategy', 'Shared layers', 'Round', 'Accuracy gain per MB', 'Alpha', 'Dataset', 'Model']].groupby(
            by=['Strategy', 'Round', 'Shared layers', 'Dataset', 'Alpha', 'Model']).apply(lambda e: comparison_with_shared_layers(df=e, df_aux=aux)).reset_index()[['Strategy', 'Round', 'Shared layers', 'Alpha', 'Accuracy (%)', 'Accuracy reduction (%)', 'Parameters reduction (MB)', 'Parameters reduction (%)', 'Dataset', 'Model']]

        df_preprocessed = copy.deepcopy(df)

        df = df[df['Shared layers'] != "100% of the layers"]
        df = df[df['Shared layers'] != "{1}"]
        # layer_selection_evaluate =  ['FedPredict (with ALS)']
        layer_selection_evaluate = ['FedPredict (with ALS + Compredict)']
        print("menor: ", df['Accuracy reduction (%)'].min())
        print("Fed", df[df['Shared layers'] == 'FedPredict (with ALS)'][['Accuracy reduction (%)', 'Round']])
        print("tra: ", df['Shared layers'].unique().tolist())

        x_column = 'Round'
        y_column = 'Accuracy reduction (%)'
        hue = 'Shared layers'
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
        hue = 'Shared layers'
        style = 'Alpha'
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
        hue = 'Shared layers'
        style = 'Alpha'
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
    num_rounds = 50
    epochs = 1
    # layer_selection_evaluate = [-1, 1, 2, 3, 4, 12, 13, 14, 123, 124, 134, 23, 24, 1234, 34]
    #layer_selection_evaluate = [1, 12, 123, 1234]
    # layer_selection_evaluate = [4, 34, 234, 1234]
    layer_selection_evaluate = [-1, 10]
    comment = "set"

    Varying_Shared_layers(tp=type_model, strategy_name=strategy, fraction_fit=fraction_fit, aggregation_method=aggregation_method, new_clients=False, new_clients_train=False, num_clients=num_clients,
                          model_name=model_name, dataset=dataset, class_per_client=2, alpha=alpha, num_rounds=num_rounds, epochs=epochs,
                          comment=comment, layer_selection_evaluate=layer_selection_evaluate).start()
