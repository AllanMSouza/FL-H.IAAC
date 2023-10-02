import numpy as np
import pandas as pd
from base_plots import bar_plot, line_plot, ecdf_plot, box_plot
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import os

class JointAnalysis():
    def __int__(self):
        pass

    def build_filenames_and_read(self, type, strategies, fractions_fit, datasets, experiments, alpha, rounds, clients='20', model='CNN', file_type='evaluate_client.csv'):

        df_concat = None
        count = 0
        for i in experiments:
            experiment = experiments[i]
            new_clients = experiment['new_client']
            new_clients_train = experiment['new_client_train']
            local_epochs = experiment['local_epochs']
            comment = experiment['comment']
            layer_selection_evaluate_list = [experiment['layer_selection_evaluate']]

            for dataset in datasets:

                for fraction_fit in fractions_fit:

                    for strategy in strategies:

                        for layer_selection_evaluate in layer_selection_evaluate_list:

                            if layer_selection_evaluate == 10 and strategy != "FedPredict":
                                continue
                            filename = """{}/{}/{}-None-{}/new_clients_{}_train_{}/{}/{}/{}/{}/alpha_{}/{}_rounds/{}/{}_comment/{}_layer_selection_evaluate/{}""".format(os.path.abspath(os.path.join(os.getcwd(),
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
                                                                                                                    layer_selection_evaluate,
                                                                                                                    file_type)

                        if strategy == "FedPredict" and layer_selection_evaluate == -2:
                            strategy = "FedAvg"
                            s = "$+FP_{dc}$"
                        elif strategy == "FedYogi_with_FedPredict" and layer_selection_evaluate == -2:
                            strategy = "FedYogi"
                            s = "$+FP_{dc}$"
                        else:
                            s = "Original"



                        df = pd.read_csv(filename)
                        df['Strategy'] = np.array([strategy] * len(df))
                        df['Version'] = np.array([s] * len(df))
                        df['Experiment'] = np.array([i] * len(df))
                        df['Fraction fit'] = np.array([fraction_fit] * len(df))
                        df['Dataset'] = np.array([dataset] * len(df))
                        df['Solution'] = np.array([layer_selection_evaluate] * len(df))
                        df['Strategy'] = np.array(['FedAvg' if i=='FedAVG' else i for i in df['Strategy'].tolist()])
                        if count == 0:
                            df_concat = df
                        else:
                            df_concat = pd.concat([df_concat, df], ignore_index=True)

                        count += 1

        df_concat = self.convert_shared_layers(df_concat)
        print(df_concat)
        df_concat['Accuracy (%)'] = df_concat['Accuracy'] * 100
        df_concat['Round (t)'] = df_concat['Round']
        # plots
        # self.joint_plot_acc_two_plots(df=df_concat, experiment=1, pocs=pocs)
        # self.joint_plot_acc_two_plots(df=df_concat, experiment=1, pocs=pocs)
        df_concat = self.convert(df_concat)
        self.joint_plot_acc_four_plots(df=df_concat, experiment=1, fractions_fit=fractions_fit)
        # self.joint_plot_acc_four_plots(df=df_concat, experiment=2, fractions_fit=fractions_fit)
        # self.joint_plot_acc_four_plots(df=df_concat, experiment=3, fractions_fit=fractions_fit)
        # self.joint_plot_acc_four_plots(df=df_concat, experiment=4, fractions_fit=fractions_fit)

        # table
        strategies = [i.replace("FedAVG", "FedAvg") for i in strategies]
        # self.joint_table(df_concat, fractions_fit, strategies, experiment=1)
        # self.joint_table(df_concat, fractions_fit, strategies, experiment=2)
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

        df['Strategy'] = np.array(strategies)

        return df

    def convert_shared_layers(self, df):

        shared_layers_list = df['Solution'].tolist()
        strategies = df['Strategy'].tolist()
        for i in range(len(shared_layers_list)):
            shared_layer = str(shared_layers_list[i])
            strategy = strategies[i]
            if "FedPredict" in strategy:
                if "-1" in shared_layer:
                    shared_layers_list[i] = "$FedPredict_{d}$"
                    continue
                elif "-2" in shared_layer:
                    shared_layers_list[i] = "$FedPredict_{dc}$"
                    continue
                elif "-3" in shared_layer:
                    shared_layers_list[i] = "$FedPredict_{c}$"
                    continue
                elif "10" in shared_layer:
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

    def joint_table(self, df, pocs, strategies, experiment):

        model_report = {i: {} for i in strategies}
        # df = df[df['Round (t)'] == 100]
        # df_test = df[['Round (t)', 'Size of parameters', 'Strategy', 'Accuracy (%)', 'Experiment', 'Fraction fit', 'Dataset']].groupby(
        #     ['Round (t)', 'Strategy', 'Experiment', 'Fraction fit', 'Dataset']).apply(
        #     lambda e: self.groupb_by_table(e)).reset_index()[
        #     ['Round (t)', 'Strategy', 'Experiment', 'Fraction fit', 'Dataset', 'Size of parameters (bytes)', 'Accuracy (%)']]
        df_test = df[
            ['Round (t)', 'Size of parameters', 'Strategy', 'Accuracy (%)', 'Experiment', 'Fraction fit', 'Dataset']]

        # df_test = df_test.query("""Round in [10, 100]""")
        print("agrupou table")
        print(df_test)
        convert_dict = {0.1: 5, 0.2: 10, 0.3: 15, 0.4: 20}
        # df_test['Fraction fit'] = np.array([convert_dict[i] for i in df_test['Fraction fit'].tolist()])

        columns = ['0.2', '0.3']

        index = [np.array(['EMNIST'] * len(columns) + ['CIFAR-10'] * len(columns)), np.array(columns * 2)]

        models_dict = {}
        ci = 0.95
        for model_name in model_report:

            mnist_acc = {}
            cifar10_acc = {}
            for column in columns:

                # mnist_acc[column] = (self.filter(df_test, experiment, 'MNIST', float(column), strategy=model_name)['Accuracy (%)']*100).mean().round(6)
                # cifar10_acc[column] = (self.filter(df_test, experiment, 'CIFAR10', float(column), strategy=model_name)['Accuracy (%)']*100).mean().round(6)
                mnist_acc[column] = self.t_distribution((self.filter(df_test, experiment, 'EMNIST', float(column), strategy=model_name)[
                                         'Accuracy (%)']).tolist(), ci)
                cifar10_acc[column] = self.t_distribution((self.filter(df_test, experiment, 'CIFAR10', float(column), strategy=model_name)[
                                           'Accuracy (%)']).tolist(), ci)

            model_metrics = []

            for column in columns:
                model_metrics.append(mnist_acc[column])
            for column in columns:
                model_metrics.append(cifar10_acc[column])

            models_dict[model_name] = model_metrics

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
        df_table.columns = np.array(['$FedPredict_{dc}$', 'FedAvg', 'FedClassAvg'])
        print(df_table.columns)

        latex = df_table.to_latex().replace("\\\nEMNIST", "\\\n\hline\nEMNIST").replace("\\\nCIFAR-10", "\\\n\hline\nCIFAR-10").replace("\\bottomrule", "\\hline\n\\bottomrule").replace("\\midrule", "\\hline\n\\midrule").replace("\\toprule", "\\hline\n\\toprule").replace("textbf", r"\textbf").replace("\}", "}").replace("\{", "{").replace("\\begin{tabular", "\\resizebox{\columnwidth}{!}{\\begin{tabular}")

        base_dir = """analysis/output/experiment_{}/""".format(str(experiment + 1))
        filename = """{}latex_{}.txt""".format(base_dir, str(experiment))
        pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

        #  df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF1-score")

    def groupb_by_plot(self, df):
        parameters = int(df['Size of parameters'].mean())
        accuracy = float(df['Accuracy (%)'].mean())
        loss = float(df['Loss'].mean())

        return pd.DataFrame({'Size of parameters (bytes)': [parameters], 'Accuracy (%)': [accuracy], 'Loss': [loss]})

    def filter(self, df, experiment, dataset, fraction_fit, strategy=None):

        # df['Accuracy (%)'] = df['Accuracy (%)']*100
        if strategy is not None:
            df = df.query(
                """Experiment=={} and Dataset=='{}' and Strategy=='{}'""".format(str(experiment), str(dataset), strategy))
            df = df[df['Fraction fit'] == fraction_fit]
        else:
            df = df.query(
                """Experiment=={} and Dataset=='{}'""".format(str(experiment), (dataset)))
            df = df[df['Fraction fit'] == fraction_fit]

        print("filtrou: ", df)

        return df

    def filter_and_plot(self, ax, base_dir, filename, title, df, experiment, dataset, fraction_fit, x_column, y_column, hue, hue_order=None, style=None):

        df = self.filter(df, experiment, dataset, fraction_fit)

        print("filtrado: ", df, df[hue].unique().tolist())
        box_plot(df=df, base_dir=base_dir, file_name=filename, x_column=x_column, y_column=y_column, title=title, hue=hue, ax=ax, tipo='1', hue_order=hue_order)

    def joint_plot_acc_four_plots(self, df, experiment, fractions_fit):
        print("Joint plot exeprimento: ", experiment)

        df = df[df['nt'].isin([1, 10])]
        df = df[df['Round (t)'] > 80]
        # df = df[df['Round (t)'] < 30]
        df['nt'] = df['nt'].astype(int)
        df_test = df[['Round (t)', 'Loss', 'Size of parameters', 'Strategy', 'Accuracy (%)', 'Experiment', 'Fraction fit', 'Dataset', 'nt']].groupby(['Round (t)', 'Strategy', 'Experiment', 'Fraction fit', 'Dataset', 'nt']).apply(lambda e: self.groupb_by_plot(e)).reset_index()[['Round (t)', 'Strategy', 'Experiment', 'Fraction fit', 'Dataset', 'Size of parameters (bytes)', 'Accuracy (%)', 'Loss', 'nt']]
        print("agrupou plot")
        print(df_test)
        # figsize=(12, 9),
        sns.set(style='whitegrid')
        fig, axs = plt.subplots(1, 2,  sharex='all', sharey='all', figsize=(6, 6))

        x_column = 'nt'
        y_column = 'Accuracy (%)'
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        base_dir = """analysis/output/experiment_{}/""".format(str(experiment+1))
        # ====================================================================
        fraction_fit = fractions_fit[0]
        dataset = 'EMNIST'
        title = """{}""".format(dataset)
        filename = 'nt'
        i = 0
        hue_order = ['$FedAvg+FP_{dc}$', 'FedAvg', '$FedYogi+FP_{dc}$', "FedYogi"]
        print(df_test['Strategy'].unique().tolist())
        hue = 'nt'
        self.filter_and_plot(ax=axs[i], base_dir=base_dir, filename=filename, title=title, df=df_test,
                             experiment=experiment, dataset=dataset, fraction_fit=fraction_fit, x_column=x_column, y_column=y_column,
                             hue='Strategy', hue_order=hue_order, style=None)
        axs[i].get_legend().remove()
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
        # ====================================================================
        fraction_fit = fractions_fit[0]
        dataset = 'CIFAR10'
        title = """CIFAR-10""".format()
        i = 1
        self.filter_and_plot(ax=axs[i], base_dir=base_dir, filename=filename, title=title, df=df_test,
                             experiment=experiment, dataset=dataset, fraction_fit=fraction_fit, x_column=x_column, y_column=y_column,
                             hue='Strategy', hue_order=hue_order, style='nt')
        axs[i].get_legend().remove()
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
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


        lines_labels = [axs[0].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.06))
        print("""{}joint_plot_four_plot_{}_nt.png""".format(base_dir, str(experiment)))
        fig.savefig("""{}joint_plot_four_plot_{}_nt_{}_alpha.png""".format(base_dir, str(experiment), alpha), bbox_inches='tight', dpi=400)
        fig.savefig("""{}joint_plot_four_plot_{}_nt_{}_alpha.svg""".format(base_dir, str(experiment), alpha), bbox_inches='tight', dpi=400)

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
         'comment': 'set', 'layer_selection_evaluate': -2, 'local_epochs': '1_local_epochs'}}

    strategies = ['FedPredict', 'FedYogi_with_FedPredict', 'FedAVG', 'FedYogi']
    # pocs = [0.1, 0.2, 0.3]
    fractions_fit = [0.3]
    # datasets = ['MNIST', 'CIFAR10']
    datasets = ['EMNIST', 'CIFAR10']
    alpha = 5.0
    rounds = 100
    clients = '20'
    model = 'CNN_3'
    type = 'torch'
    file_type = 'evaluate_client.csv'

    joint_plot = JointAnalysis()
    joint_plot.build_filenames_and_read(type, strategies, fractions_fit, datasets, experiments, alpha, rounds, clients, model, file_type)