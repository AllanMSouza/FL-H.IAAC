import numpy as np
import pandas as pd
from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import os

class JointAnalysis():
    def __int__(self):
        pass

    def build_filenames_and_read(self, type, strategies, pocs, datasets, experiments, clients='50', model='CNN', file_type='evaluate_client.csv'):

        df_concat = None
        count = 0
        for i in experiments:
            experiment = experiments[i]
            new_clients = experiment['new_clients']
            local_epochs = experiment['local_epochs']

            for dataset in datasets:

                if dataset == 'MotionSense':
                    clients = 24
                    model = 'DNN'
                    poc = 0.3
                elif dataset == 'UCIHAR':
                    clients = 30
                    model = 'DNN'
                    poc = 0.3
                else:
                    clients = 50
                    model = 'CNN'
                    poc = 0.3

                pocs = [poc]
                for poc in pocs:

                    for strategy in strategies:
                        # if strategy == 'FedPredict':
                        #     poc = 0.3
                        # else:
                        #     poc = 0.2
                        filename = """{}/{}/{}-POC-{}/{}/{}/{}/{}/{}/{}""".format(os.path.abspath(os.path.join(os.getcwd(),
                                                                                                                os.pardir)) + "/FedLTA/logs",
                                                                                                                type,
                                                                                                                strategy,
                                                                                                                poc,
                                                                                                                new_clients,
                                                                                                                clients,
                                                                                                                model,
                                                                                                                dataset,
                                                                                                                local_epochs,
                                                                                                                file_type)

                        df = pd.read_csv(filename)
                        df['Strategy'] = np.array([strategy] * len(df))
                        df['Experiment'] = np.array([i] * len(df))
                        df['POC'] = np.array([poc] * len(df))
                        df['Dataset'] = np.array([dataset] * len(df))
                        df['Strategy'] = np.array(['FedAvg' if i=='FedAVG' else i for i in df['Strategy'].tolist()])
                        if count == 0:
                            df_concat = df
                        else:
                            df_concat = pd.concat([df_concat, df], ignore_index=True)

                        count += 1
        pocs = [0.2, 0.33]
        print(df_concat)
        df_concat['Accuracy (%)'] = df_concat['Accuracy'] * 100
        df_concat['Round (t)'] = df_concat['Round']
        # plots
        self.joint_plot_acc_acc_loss_plots(df=df_concat, experiment=1, pocs=pocs, y_column='Accuracy (%)')
        self.joint_plot_acc_acc_loss_plots(df=df_concat, experiment=1, pocs=pocs, y_column='Loss')
        # self.joint_plot_acc_two_plots(df=df_concat, experiment=1, pocs=pocs)
        # self.joint_plot_acc_four_plots(df=df_concat, experiment=1, pocs=pocs)
        # self.joint_plot_acc_two_plots(df=df_concat, experiment=2, pocs=pocs)
        #self.joint_plot_acc_four_plots(df=df_concat, experiment=2, pocs=pocs)
        # self.joint_plot_acc_two_plots(df=df_concat, experiment=3, pocs=pocs)
        #self.joint_plot_acc_four_plots(df=df_concat, experiment=3, pocs=pocs)
        # self.joint_plot_acc_two_plots(df=df_concat, experiment=4, pocs=pocs)
        #self.joint_plot_acc_four_plots(df=df_concat, experiment=4, pocs=pocs)

        # table
        # self.joint_table(df_concat, pocs, strategies, experiment=1)
        # self.joint_table(df_concat, pocs, strategies, experiment=2)
        # self.joint_table(df_concat, pocs, strategies, experiment=3)
        # self.joint_table(df_concat, pocs, strategies, experiment=4)



    def groupb_by_table(self, df):
        parameters = int(df['Size of parameters'].mean())
        accuracy = df['Accuracy (%)'].mean()

        return pd.DataFrame({'Size of parameters (bytes)': [parameters], 'Accuracy (%)': [accuracy]})

    def joint_table(self, df, pocs, strategies, experiment):

        model_report = {i: {} for i in strategies}
        df_test = df[['Round (t)', 'Size of parameters', 'Strategy', 'Accuracy (%)', 'Experiment', 'POC', 'Dataset']].groupby(
            ['Round (t)', 'Strategy', 'Experiment', 'POC', 'Dataset']).apply(
            lambda e: self.groupb_by_table(e)).reset_index()[
            ['Round (t)', 'Strategy', 'Experiment', 'POC', 'Dataset', 'Size of parameters (bytes)', 'Accuracy (%)']]

        # df_test = df_test.query("""Round in [10, 100]""")
        print("agropou")
        print(df_test)
        convert_dict = {0.1: 5, 0.2: 10, 0.3: 15, 0.4: 20}
        df_test['POC'] = np.array([convert_dict[i] for i in df_test['POC'].tolist()])

        columns = ['15', '20']

        index = [np.array(['MNIST'] * len(columns) + ['CIFAR-10'] * len(columns)), np.array(columns * 2)]

        models_dict = {}
        ci = 0.95
        for model_name in model_report:

            mnist_acc = {}
            cifar10_acc = {}
            for column in columns:

                # mnist_acc[column] = (self.filter(df_test, experiment, 'MNIST', float(column), strategy=model_name)['Accuracy (%)']*100).mean().round(6)
                # cifar10_acc[column] = (self.filter(df_test, experiment, 'CIFAR10', float(column), strategy=model_name)['Accuracy (%)']*100).mean().round(6)
                mnist_acc[column] = self.t_distribution((self.filter(df_test, experiment, 'MNIST', float(column), strategy=model_name)[
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
        df_table.columns = np.array(['FedPredict', 'FedAvg', 'FedClassAvg', 'FedPer', 'FedProto'])
        print(df_table.columns)

        latex = df_table.to_latex().replace("\\\nMNIST", "\\\n\hline\nMNIST").replace("\\\nCIFAR-10", "\\\n\hline\nCIFAR-10").replace("\\bottomrule", "\\hline\n\\bottomrule").replace("\\midrule", "\\hline\n\\midrule").replace("\\toprule", "\\hline\n\\toprule").replace("textbf", r"\textbf").replace("\}", "}").replace("\{", "{").replace("\\begin{tabular", "\\resizebox{\columnwidth}{!}{\\begin{tabular}")

        base_dir = """analysis/output/experiment_{}/""".format(str(experiment + 1))
        filename = """{}latex_{}.txt""".format(base_dir, str(experiment))
        pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

        #  df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF1-score")

    def groupb_by_plot(self, df):
        parameters = int(df['Size of parameters'].mean())
        accuracy = float(df['Accuracy (%)'].mean())
        loss = float(df['Loss'].mean())

        return pd.DataFrame({'Size of parameters (bytes)': [parameters], 'Accuracy (%)': [accuracy], 'Loss': [loss]})

    def filter(self, df, experiment, dataset, poc, strategy=None):

        # df['Accuracy (%)'] = df['Accuracy (%)']*100
        if strategy is not None:
            df = df.query(
                """Experiment=={} and POC=={} and Dataset=='{}' and Strategy=='{}'""".format(str(experiment), float(poc), str(dataset), strategy))
        else:
            df = df.query(
                """Experiment=={} and Dataset=='{}'""".format(str(experiment), str(dataset)))
        print("filtrou: ", df)

        return df

    def filter_and_plot(self, ax, base_dir, filename, title, df, experiment, dataset, x_column, y_column, hue, hue_order=None, poc=None):

        df = self.filter(df, experiment, dataset, poc)

        print("filtrado: ", df, df[hue].unique().tolist())
        line_plot(df=df, base_dir=base_dir, file_name=filename, x_column=x_column, y_column=y_column, title=title, hue=hue, ax=ax, type='1', hue_order=hue_order)
    def joint_plot_acc_acc_loss_plots(self, df, experiment, pocs, y_column = 'Accuracy (%)'):
        print("Joint plot exeprimento: ", experiment)

        df_test = df[['Round (t)', 'Loss', 'Size of parameters', 'Strategy', 'Accuracy (%)', 'Experiment', 'POC',
                      'Dataset']].groupby(['Round (t)', 'Strategy', 'Experiment', 'POC', 'Dataset']).apply(
            lambda e: self.groupb_by_plot(e)).reset_index()[
            ['Round (t)', 'Strategy', 'Experiment', 'POC', 'Dataset', 'Size of parameters (bytes)', 'Accuracy (%)',
             'Loss']]
        print("agrupou")
        print(df_test)
        # figsize=(12, 9),
        sns.set(style='whitegrid')
        fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(6, 6))

        x_column = 'Round (t)'
        poc = None
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        base_dir = """analysis/output/experiment_{}/""".format(str(experiment + 1))
        solutions = {1: ['FedPredict', 'FedAvg', 'FedClassAvg', 'QFedAvg', 'FedPer', 'FedProto', 'FedYogi'], 2: ['FedClassAvg', 'FedProto']}
        # ====================================================================
        dataset = 'MNIST'
        solutions_set = 1
        title = """{}""".format(dataset+' (non-IID)')
        filename = ''
        i = 0
        j = 0
        hue_order = ['FedPredict', 'FedAvg', 'FedClassAvg', 'QFedAvg', 'FedPer', 'FedProto', 'FedYogi']
        df = df_test
        self.filter_and_plot(ax=None, base_dir=base_dir, filename=title+y_column.replace("(%)", ""), title=title, df=df,
                             experiment=experiment, dataset=dataset, x_column=x_column, y_column=y_column,
                             hue='Strategy', hue_order=hue_order, poc=poc)
        # axs[i, j].get_legend().remove()
        # axs[i, j].set_xlabel('')
        # axs[i, j].set_ylabel('')
        # ====================================================================
        dataset = 'CIFAR10'
        solutions_set = 2
        title = """{}""".format('CIFAR-10'+' (non-IID)')
        i = 0
        j = 1
        df = df_test
        self.filter_and_plot(ax=None, base_dir=base_dir, filename=title+y_column.replace("(%)", ""), title=title, df=df,
                             experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
                             hue='Strategy', hue_order=hue_order)
        # axs[i, j].get_legend().remove()
        # axs[i, j].set_xlabel('')
        # axs[i, j].set_ylabel('')
        # ====================================================================
        dataset = 'MotionSense'
        solutions_set = 1
        title = """{}""".format(dataset+' (IID)')
        i = 1
        j = 0
        df = df_test
        self.filter_and_plot(ax=None, base_dir=base_dir, filename=title+y_column.replace("(%)", ""), title=title, df=df,
                             experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
                             hue='Strategy', hue_order=hue_order)
        # axs[i, j].get_legend().remove()
        # axs[i, j].set_xlabel('')
        # axs[i, j].set_ylabel('')
        # ====================================================================
        dataset = 'UCIHAR'
        title = """{}""".format(dataset+' (IID)')
        i = 1
        j = 1
        df = df_test
        self.filter_and_plot(ax=None, base_dir=base_dir, filename=title+y_column.replace("(%)", ""), title=title, df=df,
                             experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
                             hue='Strategy', hue_order=hue_order)
        # axs[i, j].get_legend().remove()
        # axs[i, j].set_xlabel('')
        # axs[i, j].set_ylabel('')
        # ====================================================================
        # =========================///////////================================
        # fig.suptitle("", fontsize=16)
        # plt.tight_layout()
        # plt.subplots_adjust(wspace=0.07, hspace=0.14)
        # # plt.subplots_adjust(right=0.9)
        # # fig.legend(
        # #            loc="lower right")
        # # fig.legend(lines, labels)
        # # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        # fig.supxlabel(x_column, y=-0.02)
        # fig.supylabel(y_column, x=-0.01)
        #
        # lines_labels = [axs[0, 0].get_legend_handles_labels()]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # fig.legend(lines, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.10))
        # fig.savefig("""{}joint_plot_acc_loss_plot_{}_{}.png""".format(base_dir, str(experiment), y_column.replace(' (%)',"")), bbox_inches='tight', dpi=400)
        # fig.savefig("""{}joint_plot_acc_loss_plot_{}_{}.svg""".format(base_dir, str(experiment), y_column.replace(' (%)',"")), bbox_inches='tight', dpi=400)

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

    experiments = {1: {'new_clients': 'new_clients_False_train_False', 'local_epochs': '1_local_epochs'}}

    strategies = ['FedPredict', 'FedAVG', 'FedClassAvg', 'QFedAvg', 'FedPer', 'FedProto', 'FedYogi']
    # pocs = [0.1, 0.2, 0.3]
    pocs = [0.3]
    experiments = {1: experiments[1]}
    # datasets = ['MNIST', 'CIFAR10']
    datasets = ['MNIST', 'CIFAR10', 'MotionSense', 'UCIHAR']
    clients = '50'
    model = 'CNN'
    type = 'torch'
    file_type = 'evaluate_client.csv'

    joint_plot = JointAnalysis()
    joint_plot.build_filenames_and_read(type, strategies, pocs, datasets, experiments, clients, model, file_type)