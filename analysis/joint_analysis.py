import copy

import numpy as np
import pandas as pd
from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import os
import sys
from matplotlib.lines import Line2D

class JointAnalysis():
    def __int__(self):
        pass

    def build_filenames_and_read(self, type, strategies, fractions_fit, datasets, experiments, alphas, rounds, clients='20', model='CNN', file_type='evaluate_client.csv'):

        self.rounds = rounds
        self.clients = clients
        df_concat = None
        count = 0
        version_dict = {"FedPredict": {-2: "$FedPredict_{dc}"}}
        for i in experiments:
            experiment = experiments[i]
            new_clients = experiment['new_client']
            new_clients_train = experiment['new_client_train']
            local_epochs = experiment['local_epochs']
            comment = experiment['comment']
            compression_list = [experiment['compression']]

            for dataset in datasets:

                for fraction_fit in fractions_fit:

                    for strategy in strategies:

                        for compression in compression_list:

                            if compression == 10 and "FedPredict" not in strategy:
                                continue

                            # if new_clients == 'True' and alpha == 5.0 and ("FedAVG" in strategy):
                            #     continue
                            # if strategy == "FedYogi_with_FedPredict" and (not(i == 2 and dataset == "CIFAR10") and not (i == 2 and fraction_fit in [0.2, 0.3] and dataset == "EMNIST")):
                            #     continue

                            for alpha in alphas:

                                filename = """{}/{}/{}-None-{}/new_clients_{}_train_{}/{}/{}/{}/{}/alpha_{}/{}_rounds/{}/{}_comment/{}_compression/{}""".format(os.path.abspath(os.path.join(os.getcwd(),
                                                                                                                        os.pardir)) + "/FL-H.IAAC/logs",
                                                                                                                        type,
                                                                                                                        strategy,
                                                                                                                        fraction_fit,
                                                                                                                        new_clients,
                                                                                                                        new_clients_train,
                                                                                                                        clients,
                                                                                                                        model,
                                                                                                                        dataset,
                                                                                                                        "classes_per_client_2",
                                                                                                                        alpha,
                                                                                                                        rounds,
                                                                                                                        local_epochs,
                                                                                                                        comment,
                                                                                                                        compression,
                                                                                                                        file_type)

                                try:
                                    df = pd.read_csv(filename)
                                except:
                                    continue
                                if strategy == "FedPredict" and compression == "dls_compredict":
                                    st = "FedAvg"
                                    s = "$+FP_{dc}$"
                                elif strategy == "FedYogi_with_FedPredict" and compression == "dls_compredict":
                                    st = "FedYogi"
                                    s = "$+FP_{dc}$"
                                elif strategy == "FedKD_with_FedPredict" and compression == "dls_compredict":
                                    st = "FedKD"
                                    s = "$+FP_{dc}$"
                                else:
                                    st = copy.copy(strategy).replace('FedAVG', 'FedAvg')
                                    s = "$Original$"


                                df['Strategy'] = np.array([st] * len(df))
                                df['Version'] = np.array([s] * len(df))
                                df['Experiment'] = np.array([i] * len(df))
                                df['Fraction fit'] = np.array([fraction_fit] * len(df))
                                df['Dataset'] = np.array([dataset] * len(df))
                                df['Solution'] = np.array([compression] * len(df))
                                df['Alpha'] = np.array([alpha] * len(df))
                                if count == 0:
                                    df_concat = df
                                else:
                                    df_concat = pd.concat([df_concat, df], ignore_index=True)

                                count += 1

        # df_concat = self.convert_shared_layers(df_concat)
        df_concat['Accuracy (%)'] = df_concat['Accuracy'] * 100
        df_concat['Round (t)'] = df_concat['Round']
        # plots
        # self.joint_plot_acc_two_plots(df=df_concat, experiment=1, pocs=pocs)
        # self.joint_plot_acc_two_plots(df=df_concat, experiment=1, pocs=pocs)

        self.joint_plot_acc_four_plots(df=df_concat, experiment=1, alphas=alphas)
        # self.joint_plot_acc_four_plots(df=df_concat, experiment=2, alphas=alphas)
        # self.joint_plot_acc_four_plots(df=df_concat, experiment=3, alphas=alphas)
        # self.joint_plot_acc_four_plots(df=df_concat, experiment=4, fractions_fit=fractions_fit)
        # table




        print("estrategias: ", strategies)
        print("versao: ", df_concat[['Strategy', 'Version']].drop_duplicates())
        df_concat = self.convert(df_concat)
        strategies = df_concat['Strategy'].unique().tolist()
        print("versao2: ", df_concat['Strategy'].unique().tolist())
        aux = []
        order = ['$FedAvg+FP_{dc}$', '$FedAvg$', '$FedYogi+FP_{dc}$', '$FedYogi$', '$FedKD+FP_{dc}$', '$FedKD$', '$FedPer$', '$FedProto$']
        for s in order:
            if s in strategies:
                aux.append(s)
        strategies = aux
        print("finai: ", strategies)
        print("Experimento 1")
        self.joint_table(df_concat, alphas, strategies, experiment=1)
        print("Experimento 2")
        # self.joint_table(df_concat, alphas, strategies, experiment=2)
        # self.joint_table(df_concat, fractions_fit, strategies, experiment=3)
        # self.joint_table(df_concat, fractions_fit, strategies, experiment=4)

    def convert(self, df):

        versions = df['Version'].tolist()
        strategies = df['Strategy'].tolist()

        for i in range(len(versions)):
            version = versions[i]
            strategy = strategies[i]

            if version == "$+FP_{dc}$":
                strategy = "$" + strategy + "+FP_{dc}$"
                strategies[i] = strategy
            else:
                strategy = "$" + strategy + "$"
                strategies[i] = strategy

        df['Strategy'] = np.array(strategies)

        return df

    def convert_shared_layers(self, df):

        shared_layers_list = df['Solution'].tolist()
        strategies = df['Strategy'].tolist()
        for i in range(len(shared_layers_list)):
            shared_layer = str(shared_layers_list[i])
            strategy = strategies[i]
            if "FedPredict" in strategy:
                if "dls" in shared_layer:
                    shared_layers_list[i] = "$FedPredict_{d}$"
                    continue
                elif "dls_compredict" in shared_layer:
                    shared_layers_list[i] = "$FedPredict_{dc}$"
                    continue
                elif "compredict" in shared_layer:
                    shared_layers_list[i] = "$FedPredict_{c}$"
                    continue
                elif "no" in shared_layer:
                    shared_layers_list[i] = "$FedPredict$"
                    continue
            else:
                shared_layers_list[i] = strategy

        df['Strategy'] = np.array(shared_layers_list)

        return df

    def groupb_by_table(self, df):
        parameters = int(df['Size of parameters'].mean())
        accuracy = df['Accuracy (%)'].mean()

        return pd.DataFrame({'Size of parameters (bytes)': [parameters], 'Accuracy (%)': [accuracy]})

    def accuracy_improvement(self, df):

        df_difference = copy.deepcopy(df)
        columns = df.columns.tolist()
        indexes = df.index.tolist()

        datasets = ['EMNIST', 'CIFAR-10', 'GTSRB']
        solutions = pd.Series([i[1] for i in indexes]).unique().tolist()
        reference_solutions = {}
        for solution_key in solutions:
            if "FP_{dc}" in solution_key:
                reference_solutions[solution_key] = solution_key.replace("+FP_{dc}", "")

        # print("indexes: ", indexes)
        # print("reference: ", reference_solutions)

        for dataset in datasets:
            for solution in reference_solutions:
                reference_index = (dataset, solution)
                target_index = (dataset, reference_solutions[solution])

                for column in columns:
                    difference = str(round(float(df.loc[reference_index, column][:4]) - float(df.loc[target_index, column][:4]), 1))
                    difference = str(round(float(difference)*100/float(df.loc[target_index, column][:4]), 1))
                    if difference[0] != "-":
                        difference = "textuparrow" + difference
                    else:
                        difference = "textdownarrow" + difference.replace("-", "")
                    df_difference.loc[reference_index, column] = "(" + difference + "%)" + df.loc[reference_index, column]


        # print(indexes)
        # print(indexes[0])
        # print(df_difference)

        return df_difference


    def joint_table(self, df, pocs, strategies, experiment):



        model_report = {i: {} for i in df['Alpha'].unique().tolist()}
        if experiment == 1:
            df = df[df['Round (t)'] == 100]
        # df_test = df[['Round (t)', 'Size of parameters', 'Strategy', 'Accuracy (%)', 'Experiment', 'Fraction fit', 'Dataset']].groupby(
        #     ['Round (t)', 'Strategy', 'Experiment', 'Fraction fit', 'Dataset']).apply(
        #     lambda e: self.groupb_by_table(e)).reset_index()[
        #     ['Round (t)', 'Strategy', 'Experiment', 'Fraction fit', 'Dataset', 'Size of parameters (bytes)', 'Accuracy (%)']]
        df_test = df[
            ['Round (t)', 'Size of parameters', 'Strategy', 'Accuracy (%)', 'Experiment', 'Fraction fit', 'Dataset', 'Alpha']]

        # df_test = df_test.query("""Round in [10, 100]""")
        print("agrupou table")
        print(df_test)
        convert_dict = {0.1: 5, 0.2: 10, 0.3: 15, 0.4: 20}
        # df_test['Fraction fit'] = np.array([convert_dict[i] for i in df_test['Fraction fit'].tolist()])

        columns = strategies

        index = [np.array(['EMNIST'] * len(columns) +['CIFAR-10'] * len(columns) + ['GTSRB'] * len(columns)), np.array(columns * 3)]

        models_dict = {}
        ci = 0.95
        print("filtro tabela")
        for model_name in model_report:

            mnist_acc = {}
            cifar10_acc = {}
            gtsrb_acc = {}
            for column in columns:

                # mnist_acc[column] = (self.filter(df_test, experiment, 'MNIST', float(column), strategy=model_name)['Accuracy (%)']*100).mean().round(6)
                # cifar10_acc[column] = (self.filter(df_test, experiment, 'CIFAR10', float(column), strategy=model_name)['Accuracy (%)']*100).mean().round(6)
                mnist_acc[column] = self.t_distribution((self.filter(df_test, experiment, 'EMNIST', float(model_name), strategy=column)[
                                         'Accuracy (%)']).tolist(), ci)
                cifar10_acc[column] = self.t_distribution((self.filter(df_test, experiment, 'CIFAR10', float(model_name), strategy=column)[
                                           'Accuracy (%)']).tolist(), ci)
                gtsrb_acc[column] = self.t_distribution(
                    (self.filter(df_test, experiment, 'GTSRB', float(model_name), strategy=column)[
                        'Accuracy (%)']).tolist(), ci)

            model_metrics = []

            for column in columns:
                model_metrics.append(mnist_acc[column])
            for column in columns:
                model_metrics.append(cifar10_acc[column])
            for column in columns:
                model_metrics.append(gtsrb_acc[column])

            models_dict[model_name] = model_metrics

        df_table = pd.DataFrame(models_dict, index=index).round(4)
        print("df table: ", df_table)
        print(df_table.to_string())

        df_accuracy_improvements = self.accuracy_improvement(df_table)

        indexes = df_table.index.tolist()
        n_solutions = len(pd.Series([i[1] for i in indexes]).unique().tolist())
        max_values = self.idmax(df_table, n_solutions)
        print("max values", max_values)

        for max_value in max_values:
            row_index = max_value[0]
            column = max_value[1]
            column_values = df_accuracy_improvements[column].tolist()
            column_values[row_index] = "textbf{" + str(column_values[row_index]) + "}"

            df_accuracy_improvements[column] = np.array(column_values)

        df_accuracy_improvements.columns = np.array(list(model_report.keys()))
        print("melhorias")
        print(df_accuracy_improvements)

        indexes = [0.1, 1.0]
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

        latex = df_accuracy_improvements.to_latex().replace("\\\nEMNIST", "\\\n\hline\nEMNIST").replace("\\\nGTSRB", "\\\n\hline\nGTSRB").replace("\\bottomrule", "\\hline\n\\bottomrule").replace("\\midrule", "\\hline\n\\midrule").replace("\\toprule", "\\hline\n\\toprule").replace("textbf", r"\textbf").replace("\}", "}").replace("\{", "{").replace("\\begin{tabular", "\\resizebox{\columnwidth}{!}{\\begin{tabular}").replace("\$", "$").replace("textuparrow", "\oitextuparrow").replace("textdownarrow", "\oitextdownarrow").replace("\&", "&").replace("\_", "_")

        base_dir = """analysis/output/experiment_{}/""".format(str(experiment + 1))
        filename = """{}latex_{}.txt""".format(base_dir, str(experiment))
        pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

        self.improvements(df_table, experiment + 1)

        #  df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF1-score")



    def improvements(self, df, experiment):

        strategies = {"$FedAvg+FP_{dc}$": "$FedAvg$", "$FedYogi+FP_{dc}$": "$FedYogi$"}
        datasets = ['EMNIST', 'GTSRB']
        columns = df.columns.tolist()
        improvements_dict = {'Dataset': [], 'Strategy': [], 'Original strategy': [], 'Alpha': [], 'Accuracy (%)': []}
        df_improvements = pd.DataFrame(improvements_dict)

        for dataset in datasets:
            for strategy in strategies:
                original_strategy = strategies[strategy]

                for j in range(len(columns)):

                    index = (dataset, strategy)
                    index_original = (dataset, original_strategy)

                    acc = float(df.loc[index].tolist()[j].replace("textbf{", "")[:4])
                    acc_original = float(df.loc[index_original].tolist()[j].replace("textbf{", "")[:4])

                    row = {'Dataset': dataset, 'Strategy': strategy, 'Original strategy': original_strategy, 'Alpha': columns[j], 'Accuracy (%)': acc - acc_original}

                    df_improvements = df_improvements.append(row, ignore_index=True)

        print("Experiment: ", experiment + 1)
        print(df_improvements)

    def groupb_by_plot(self, df):
        parameters = int(df['Size of parameters'].mean())
        accuracy = float(df['Accuracy (%)'].mean())
        loss = float(df['Loss'].mean())

        return pd.DataFrame({'Size of parameters (bytes)': [parameters], 'Accuracy (%)': [accuracy], 'Loss': [loss]})

    def filter(self, df, experiment, dataset, alpha, strategy=None):

        # df['Accuracy (%)'] = df['Accuracy (%)']*100
        if strategy is not None:
            df = df.query(
                """Experiment=={} and Dataset=='{}' and Strategy=='{}'""".format(str(experiment), str(dataset), strategy))
            df = df[df['Alpha'] == alpha]
        else:
            df = df.query(
                """Experiment=={} and Dataset=='{}'""".format(str(experiment), (dataset)))
            df = df[df['Alpha'] == alpha]

        print("filtrou: ", df, experiment, dataset, alpha, strategy)

        return df

    def filter_and_plot(self, ax, base_dir, filename, title, df, experiment, dataset, alpha, x_column, y_column, hue, hue_order=None, style=None, markers=None, size=None, sizes=None):

        df = self.filter(df, experiment, dataset, alpha)
        df['Strategy'] = np.array(["$" + i + "$" for i in df['Strategy'].tolist()])

        print("filtrado: ", df, df[hue].unique().tolist())
        line_plot(df=df, base_dir=base_dir, file_name=filename, x_column=x_column, y_column=y_column, title=title, hue=hue, ax=ax, tipo='1', hue_order=hue_order, style=style, markers=markers, size=size, sizes=sizes)

    def joint_plot_acc_four_plots(self, df, experiment, alphas):
        print("Joint plot exeprimento: ", experiment)

        df_test = df[['Round (t)', 'Loss', 'Size of parameters', 'Strategy', 'Accuracy (%)', 'Experiment', 'Fraction fit', 'Dataset', 'Version', 'Alpha']].groupby(['Round (t)', 'Strategy', 'Experiment', 'Fraction fit', 'Dataset', 'Version', 'Alpha']).apply(lambda e: self.groupb_by_plot(e)).reset_index()[['Round (t)', 'Strategy', 'Experiment', 'Fraction fit', 'Dataset', 'Size of parameters (bytes)', 'Accuracy (%)', 'Loss', 'Version', 'Alpha']]
        print("agrupou plot")
        print(df_test[df_test['Round (t)']==100])
        # figsize=(12, 9),
        sns.set(style='whitegrid')
        fig, axs = plt.subplots(2, 3,  sharex='all', sharey='all', figsize=(9, 6))

        x_column = 'Round (t)'
        y_column = 'Accuracy (%)'
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        base_dir = """analysis/output/experiment_{}/""".format(str(experiment+1))
        # ====================================================================
        alpha = alphas[0]
        dataset = 'EMNIST'
        title = """{}; \u03B1={}""".format(dataset, alpha)
        filename = ''
        i = 0
        j = 0
        # hue_order = ['$FedPredict_{dc}$', "$FedPredict$", 'FedClassAvg', 'FedAvg']
        hue_order = ['$FedAvg$', '$FedYogi$', '$FedKD$', '$FedPer$', '$FedProto$']
        style = "Version"
        # markers = [',', '.'
        markers = None
        size = None
        sizes = (1, 1.8)
        self.filter_and_plot(ax=axs[i,j], base_dir=base_dir, filename=filename, title=title, df=df_test,
                             experiment=experiment, dataset=dataset, alpha=alpha, x_column=x_column, y_column=y_column,
                             hue='Strategy', hue_order=hue_order, style=style, markers=markers, size=size, sizes=sizes)
        # axs[i,j].get_legend().remove()
        axs[i,j].legend(fontsize=7)
        axs[i,j].set_xlabel('')
        axs[i,j].set_ylabel('')
        # sort both labels and handles by labels
        # order = ['FedPredict', 'FedClassAvg', 'FedPer', 'FedAvg', 'FedProto']
        # order = {k: f for k, f in zip(order, handles)}

        # handles, labels = plt.gca().get_legend_handles_labels()
        #
        # # specify order of items in legend
        # order = [1, 2, 0, 4, 3]
        #
        # # add legend to plot
        # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        # ====================================================================
        alpha = alphas[1]
        dataset = 'EMNIST'
        title = """{}; \u03B1={}""".format(dataset, float(alpha))
        i = 1
        j = 0
        self.filter_and_plot(ax=axs[i,j], base_dir=base_dir, filename=filename, title=title, df=df_test,
                             experiment=experiment, dataset=dataset, alpha=alpha, x_column=x_column, y_column=y_column,
                             hue='Strategy', hue_order=hue_order, style=style, markers=markers, size=size, sizes=sizes)
        axs[i,j].get_legend().remove()
        axs[i, j].set_xlabel('')
        axs[i, j].set_ylabel('')

        # ====================================================================
        alpha = alphas[0]
        dataset = 'CIFAR10'
        title = """{}; \u03B1={}""".format(dataset.replace("CIFAR10", "CIFAR-10"), float(alpha))
        i = 0
        j = 1
        self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
                             experiment=experiment, dataset=dataset, alpha=alpha, x_column=x_column, y_column=y_column,
                             hue='Strategy', hue_order=hue_order, style=style, markers=markers, size=size, sizes=sizes)
        axs[i, j].get_legend().remove()
        # axs[i].set_xlabel('')
        # axs[i].set_ylabel('')
        # ====================================================================
        alpha = alphas[1]
        dataset = 'CIFAR10'
        title = """CIFAR-10; \u03B1={}""".format(float(alpha))
        i = 1
        j = 1
        self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
                             experiment=experiment, dataset=dataset, alpha=alpha, x_column=x_column, y_column=y_column,
                             hue='Strategy', hue_order=hue_order, style=style, markers=markers, size=size, sizes=sizes)
        axs[i, j].get_legend().remove()
        axs[i, j].set_xlabel('')
        axs[i, j].set_ylabel('')
        # ====================================================================
        alpha = alphas[0]
        dataset = 'GTSRB'
        title = """GTSRB; \u03B1={}""".format(alpha)
        i = 0
        j = 2
        self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
                             experiment=experiment, dataset=dataset, alpha=alpha, x_column=x_column, y_column=y_column,
                             hue='Strategy', hue_order=hue_order, style=style, markers=markers, size=size, sizes=sizes)
        axs[i, j].get_legend().remove()
        axs[i, j].set_xlabel('')
        axs[i, j].set_ylabel('')
        # ====================================================================
        alpha = alphas[1]
        dataset = 'GTSRB'
        title = """GTSRB; \u03B1={}""".format(alpha)
        i = 1
        j = 2
        self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
                             experiment=experiment, dataset=dataset, alpha=alpha, x_column=x_column, y_column=y_column,
                             hue='Strategy', hue_order=hue_order, style=style, markers=markers, size=size, sizes=sizes)
        axs[i, j].get_legend().remove()
        axs[i, j].set_xlabel('')
        axs[i, j].set_ylabel('')
        # ====================================================================
        # poc = pocs[2]
        # dataset = 'CIFAR10'
        # title = """CIFAR-10 ({})""".format(poc)
        # i = 1
        # j = 1
        # self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
        #                      experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
        #                      hue='Strategy')
        # axs[i, j].get_legend().remove()
        # axs[i, j].set_xlabel('')
        # axs[i, j].set_ylabel('')
        # ====================================================================
        # poc = pocs[2]
        # dataset = 'CIFAR10'
        # title = """CIFAR-10 ({})""".format(poc)
        # i = 1
        # j = 1
        # self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
        #                      experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
        #                      hue='Strategy')
        # legend = axs[i, j].get_legend()
        # print("legenda: ", legend)
        # # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        # # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # axs[i, j].get_legend().remove()
        # axs[i, j].set_xlabel('')
        # axs[i, j].set_ylabel('')
        # =========================///////////================================
        fig.suptitle("", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.07, hspace=0.14)
        # plt.subplots_adjust(right=0.9)
        # fig.legend(
        #            loc="lower right")
        # fig.legend(lines, labels)
        # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        fig.supxlabel(x_column, y=-0.02)
        fig.supylabel(y_column, x=-0.01)


        lines_labels = [axs[0, 0].get_legend_handles_labels()]
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
        handles = [f("o", colors[i]) for i in range(len(hue_order) + 1)]
        handles += [plt.Line2D([], [], linestyle=markers[i], color="k") for i in range(3)]
        axs[0, 0].legend(handles, labels, fontsize=7)
        print("base: ", base_dir, """{}joint_plot_four_plot_{}_{}_rounds_{}_clients.png""".format(base_dir, str(experiment), self.rounds, self.clients))
        fig.savefig("""{}joint_plot_four_plot_{}_{}_rounds_{}_clients.png""".format(base_dir, str(experiment), self.rounds, self.clients), bbox_inches='tight', dpi=400)
        fig.savefig("""{}joint_plot_four_plot_{}_{}_rounds_{}_clients.svg""".format(base_dir, str(experiment), self.rounds, self.clients), bbox_inches='tight', dpi=400)

    def idmax(self, df, n_solutions):

        df_indexes = []
        columns = df.columns.tolist()
        print("colunas", columns)
        for i in range(len(columns)):

            column = columns[i]
            column_values = df[column].tolist()
            print("ddd", column_values)
            indexes = self.select_mean(i, column_values, columns, n_solutions)
            df_indexes += indexes

        return df_indexes

    def select_mean(self, index, column_values, columns, n_solutions):

        list_of_means = []
        indexes = []
        print("ola: ", column_values, "ola0")

        for i in range(len(column_values)):

            print("valor: ", column_values[i])
            value = float(str(str(column_values[i])[:4]).replace(u"\u00B1", ""))
            interval = float(str(column_values[i])[5:8])
            minimum = value - interval
            maximum = value + interval
            list_of_means.append((value, minimum, maximum))


        for i in range(0, len(list_of_means), n_solutions):

            dataset_values = list_of_means[i: i+n_solutions]
            max_tuple = max(dataset_values, key=lambda e: e[0])
            column_min_value = max_tuple[1]
            column_max_value = max_tuple[2]
            print("maximo: ", column_max_value)
            for j in range(len(list_of_means)):
                value_tuple = list_of_means[j]
                min_value = value_tuple[1]
                max_value = value_tuple[2]
                if j >= i and j < i+n_solutions:
                    if not(max_value < column_min_value or min_value > column_max_value):
                        indexes.append([j, columns[index]])

        return indexes

    def t_distribution(self, data, ci):

        min_ = st.t.interval(alpha=ci, df=len(data) - 1,
                      loc=np.mean(data),
                      scale=st.sem(data))[0]

        mean = np.mean(data)
        average_variation = (mean - min_).round(1)
        mean = mean.round(1)

        return str(mean) + u"\u00B1" + str(average_variation)

if __name__ == '__main__':
    """
        This code generates a joint plot (multiples plots in one image) and a table of accuracy.
        It is done for each experiment.
    """

    # experiments = {1: {'new_clients': 'new_clients_False_train_False', 'local_epochs': '1_local_epochs'},
    #               2: {'new_clients': 'new_clients_True_train_False', 'local_epochs': '1_local_epochs'},
    #               3: {'new_clients': 'new_clients_True_train_True', 'local_epochs': '1_local_epochs'},
    #               4: {'new_clients': 'new_clients_True_train_True', 'local_epochs': '2_local_epochs'}}
    experiments = {1: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression': 'dls_compredict', 'local_epochs': '1_local_epochs'},
                   2: {'algorithm': 'None', 'new_client': 'True', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression': 'dls_compredict', 'local_epochs': '1_local_epochs'}}

    strategies = ['FedAVG', 'FedPredict', 'FedYogi', 'FedYogi_with_FedPredict', 'FedKD', 'FedKD_with_FedPredict', 'FedPer', 'FedProto']
    # 'FedPredict', 'FedYogi_with_FedPredict', 'FedKD_with_FedPredict', 'FedAVG', 'FedYogi', 'FedPer', 'FedProto', 'FedKD'
    # pocs = [0.1, 0.2, 0.3]
    fractions_fit = [0.3]
    # datasets = ['MNIST', 'CIFAR10']
    datasets = ['EMNIST', 'CIFAR10', 'GTSRB']
    alpha = [0.1, 1.0]
    rounds = 100
    clients = '20'
    model = 'CNN_3'
    type_t = 'torch'
    file_type = 'evaluate_client.csv'

    joint_plot = JointAnalysis()
    joint_plot.build_filenames_and_read(type_t, strategies, fractions_fit, datasets, experiments, alpha, rounds, clients, model, file_type)