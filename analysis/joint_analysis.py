import numpy as np
import pandas as pd
from t_distribution import t_distribution_test
from base_plots import bar_plot, line_plot
import matplotlib.pyplot as plt
import seaborn as sns
import os

class JointAnalysis():
    def __int__(self):
        pass

    def build_filenames_and_read(self, strategies, pocs, datasets, experiments, clients='50', model='CNN', file_type='evaluate_client.csv'):

        df_concat = None
        count = 0
        for i in experiments:
            experiment = experiments[i]
            new_clients = experiment['new_clients']
            local_epochs = experiment['local_epochs']

            for dataset in datasets:

                for poc in pocs:

                    for strategy in strategies:

                        filename = """{}/{}-POC-{}/{}/{}/{}/{}/{}/{}""".format(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/FedLTA/logs", strategy, poc, new_clients, clients, model, dataset, local_epochs, file_type)

                        df = pd.read_csv(filename)
                        df['Strategy'] = np.array([strategy] * len(df))
                        df['Experiment'] = np.array([i] * len(df))
                        df['POC'] = np.array([poc] * len(df))
                        df['Dataset'] = np.array([dataset] * len(df))
                        if count == 0:
                            df_concat = df
                        else:
                            df_concat = pd.concat([df_concat, df], ignore_index=True)

                        count += 1
        pocs = [0.2, 0.3, 0.4]
        print(df_concat)
        # plots
        # self.joint_plot(df=df_concat, experiment=1, pocs=pocs)
        # self.joint_plot(df=df_concat, experiment=2, pocs=pocs)
        # self.joint_plot(df=df_concat, experiment=3, pocs=pocs)
        # self.joint_plot(df=df_concat, experiment=4, pocs=pocs)

        # table
        self.joint_table(df_concat, pocs, strategies, experiment=1)



    def groupb_by_table(self, df):
        parameters = int(df['Size of parameters'].mean())
        # accuracy = t_distribution_test(df['Accuracy'].tolist())
        accuracy = df['Accuracy'].mean()

        return pd.DataFrame({'Size of parameters (bytes)': [parameters], 'Accuracy': [accuracy]})

    def joint_table(self, df, pocs, strategies, experiment):

        model_report = {i: {} for i in strategies}
        df_test = df[['Round', 'Size of parameters', 'Strategy', 'Accuracy', 'Experiment', 'POC', 'Dataset']].groupby(
            ['Round', 'Strategy', 'Experiment', 'POC', 'Dataset']).apply(
            lambda e: self.groupb_by_table(e)).reset_index()[
            ['Round', 'Strategy', 'Experiment', 'POC', 'Dataset', 'Size of parameters (bytes)', 'Accuracy']]

        df_test = df_test.query("""Round in [10, 100]""")

        columns = ['0.3', '0.4']

        index = [np.array(['MNIST'] * len(columns) + ['CIFAR-10'] * len(columns)), np.array(columns * 2)]

        models_dict = {}
        for model_name in model_report:

            mnist_acc = {}
            cifar10_acc = {}
            for column in columns:

                mnist_acc[column] = self.filter(df_test, experiment, 'MNIST', float(column))['Accuracy']
                cifar10_acc[column] = self.filter(df_test, experiment, 'CIFAR10', float(column))['Accuracy']

            model_metrics = []

            for column in columns:
                model_metrics.append(mnist_acc[column])
            for column in columns:
                model_metrics.append(cifar10_acc[column])

            models_dict[model_name] = model_metrics

        df_table = pd.DataFrame(models_dict, index=index).round(4)

        print(df_table)



        #  df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF1-score")

    def groupb_by_plot(self, df):
        parameters = int(df['Size of parameters'].mean())
        accuracy = float(df['Accuracy'].mean())

        return pd.DataFrame({'Size of parameters (bytes)': [parameters], 'Accuracy': [accuracy]})

    def filter(self, df, experiment, dataset, poc):

        df = df.query(
            """Experiment=={} and POC=={} and Dataset=='{}'""".format(str(experiment), str(poc), str(dataset)))

        return df

    def filter_and_plot(self, ax, base_dir, filename, title, df, experiment, dataset, poc, x_column, y_column, hue):

        df = self.filter(df, experiment, dataset, poc)

        print("filtrado: ", df, df[hue].unique().tolist())
        line_plot(df=df, base_dir=base_dir, file_name=filename, x_column=x_column, y_column=y_column, title=title, hue=hue, ax=ax, type='1')
    def joint_plot(self, df, experiment, pocs):
        print("Joint plot exeprimento: ", experiment)

        df_test = df[['Round', 'Size of parameters', 'Strategy', 'Accuracy', 'Experiment', 'POC', 'Dataset']].groupby(['Round', 'Strategy', 'Experiment', 'POC', 'Dataset']).apply(lambda e: self.groupb_by_plot(e)).reset_index()[['Round', 'Strategy', 'Experiment', 'POC', 'Dataset', 'Size of parameters (bytes)', 'Accuracy']]
        print("agrupou")
        print(df_test)
        # figsize=(12, 9),
        sns.set(style='whitegrid')
        fig, axs = plt.subplots(2, 1,  sharex='all', sharey='all', figsize=(6, 8.8))

        x_column = 'Round'
        y_column = 'Accuracy'
        base_dir = """analysis/output/experiment_{}/""".format(str(experiment+1))
        # ====================================================================
        poc = pocs[1]
        dataset = 'MNIST'
        title = dataset
        filename = ''
        i = 0
        j = 0
        self.filter_and_plot(ax=axs[i], base_dir=base_dir, filename=filename, title=title, df=df_test, experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column, hue='Strategy')
        # axs[i].get_legend().remove()
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
        # ====================================================================
        poc = pocs[1]
        dataset = 'CIFAR10'
        title = 'CIFAR-10'
        i = 1
        j = 1
        self.filter_and_plot(ax=axs[i], base_dir=base_dir, filename=filename, title=title, df=df_test, experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column, hue='Strategy')
        axs[i].get_legend().remove()
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
        # # ====================================================================
        # poc = pocs[2]
        # dataset = 'MNIST'
        # title = """MNIST ({})""".format(poc)
        # i = 0
        # j = 2
        # self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
        #                      experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
        #                      hue='Strategy')
        # axs[i, j].get_legend().remove()
        # axs[i, j].set_xlabel('')
        # axs[i, j].set_ylabel('')
        # # ====================================================================
        # poc = pocs[0]
        # dataset = 'CIFAR10'
        # title = """CIFAR-10 ({})""".format(poc)
        # i = 1
        # j = 0
        # self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
        #                      experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
        #                      hue='Strategy')
        # axs[i, j].get_legend().remove()
        # axs[i, j].set_xlabel('')
        # axs[i, j].set_ylabel('')
        # # ====================================================================
        # poc = pocs[1]
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
        # # ====================================================================
        # poc = pocs[2]
        # dataset = 'CIFAR10'
        # title = """CIFAR-10 ({})""".format(poc)
        # i = 1
        # j = 2
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
        fig.suptitle("""Exp. {}""".format(str(experiment)), fontsize=16)
        plt.tight_layout()
        # plt.subplots_adjust(right=0.9)
        # fig.legend(
        #            loc="lower right")
        # fig.legend(lines, labels)
        # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        # plt.xlabel(x_column)
        # plt.ylabel(y_column)
        fig.savefig("""{}joint_plot_{}.png""".format(base_dir, str(experiment)), bbox_inches='tight', dpi=400)
        fig.savefig("""{}joint_plot_{}.svg""".format(base_dir, str(experiment)), bbox_inches='tight', dpi=400)


if __name__ == '__main__':

    experiments = {1: {'new_clients': 'new_clients_False_train_False', 'local_epochs': '1_local_epochs'},
                  2: {'new_clients': 'new_clients_True_train_False', 'local_epochs': '1_local_epochs'},
                  3: {'new_clients': 'new_clients_True_train_True', 'local_epochs': '1_local_epochs'},
                  4: {'new_clients': 'new_clients_True_train_True', 'local_epochs': '1_local_epochs'}}

    strategies = ['FedPredict', 'FedAVG', 'FedClassAvg', 'FedPer', 'FedProto']
    # pocs = [0.1, 0.2, 0.3]
    pocs = [0.2, 0.3, 0.4]
    # datasets = ['MNIST', 'CIFAR10']
    datasets = ['MNIST', 'CIFAR10']
    clients = '50'
    model = 'CNN'
    file_type = 'evaluate_client.csv'

    joint_plot = JointAnalysis()
    joint_plot.build_filenames_and_read(strategies, pocs, datasets, experiments, clients, model, file_type)