import numpy as np
import pandas as pd
from t_distribution import T_Distribution
from base_plots import bar_plot, line_plot
import matplotlib.pyplot as plt
import os

class JointAnalysis():
    def __int__(self):
        pass

    def build_filenames_and_read(self, strategies, pocs, datasets, experiments, clients='50', model='CNN', file_type='evaluate_client.csv'):

        df_concat = None
        i = 0
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
                        if i == 0:
                            df_concat = df
                        else:
                            df_concat = pd.concat([df_concat, df], ignore_index=True)

                        i += 1
        pocs = [0.2, 0.3, 0.4]
        print(df_concat)
        self.joint_plot(df=df_concat, experiment=1, pocs=pocs)
        # self.joint_plot(df=df_concat, experiment=2, pocs=pocs)
        # self.joint_plot(df=df_concat, experiment=3, pocs=pocs)
        # self.joint_plot(df=df_concat, experiment=4, pocs=pocs)

    def groupb_by(self, df):
        parameters = int(df['Size of parameters'].mean())
        accuracy = float(df['Accuracy'].mean())

        return pd.DataFrame({'Size of parameters (bytes)': [parameters], 'Accuracy': [accuracy]})

    def filter_and_plot(self, ax, base_dir, filename, title, df, experiment, dataset, poc, x_column, y_column, hue):

        df = df.query("""Experiment=={} and POC=={} and Dataset=='{}'""".format(str(experiment), str(poc), str(dataset)))

        print("filtrado: ", df, df[hue].unique().tolist())
        line_plot(df=df, base_dir=base_dir, file_name=filename, x_column=x_column, y_column=y_column, title=title, hue=hue, ax=ax, type='1')
    def joint_plot(self, df, experiment, pocs):
        print("Joint plot exeprimento: ", experiment)

        df_test = df[['Round', 'Size of parameters', 'Strategy', 'Accuracy', 'Experiment', 'POC', 'Dataset']].groupby(['Round', 'Strategy', 'Experiment', 'POC', 'Dataset']).apply(lambda e: self.groupb_by(e)).reset_index()[['Round', 'Strategy', 'Experiment', 'POC', 'Dataset', 'Size of parameters (bytes)', 'Accuracy']]
        print("agrupou")
        print(df_test)
        # figsize=(12, 9),
        fig, axs = plt.subplots(2, 3,  sharex='all', sharey='all')

        x_column = 'Round'
        y_column = 'Accuracy'
        base_dir = """analysis/output/experiment_{}/""".format(str(experiment+1))
        # ====================================================================
        poc = pocs[0]
        dataset = 'MNIST'
        title = """MNIST ({})""".format(poc)
        filename = ''
        i = 0
        j = 0
        self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test, experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column, hue='Strategy')
        axs[i, j].get_legend().remove()
        axs[i, j].set_xlabel('')
        axs[i, j].set_ylabel('')
        # ====================================================================
        poc = pocs[1]
        dataset = 'MNIST'
        title = """MNIST ({})""".format(poc)
        i = 0
        j = 1
        self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test, experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column, hue='Strategy')
        axs[i, j].get_legend().remove()
        axs[i, j].set_xlabel('')
        axs[i, j].set_ylabel('')
        # ====================================================================
        poc = pocs[2]
        dataset = 'MNIST'
        title = """MNIST ({})""".format(poc)
        i = 0
        j = 2
        self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
                             experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
                             hue='Strategy')
        axs[i, j].get_legend().remove()
        axs[i, j].set_xlabel('')
        axs[i, j].set_ylabel('')
        # ====================================================================
        poc = pocs[0]
        dataset = 'CIFAR10'
        title = """CIFAR-10 ({})""".format(poc)
        i = 1
        j = 0
        self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
                             experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
                             hue='Strategy')
        axs[i, j].get_legend().remove()
        axs[i, j].set_xlabel('')
        axs[i, j].set_ylabel('')
        # ====================================================================
        poc = pocs[1]
        dataset = 'CIFAR10'
        title = """CIFAR-10 ({})""".format(poc)
        i = 1
        j = 1
        self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
                             experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
                             hue='Strategy')
        axs[i, j].get_legend().remove()
        axs[i, j].set_xlabel('')
        axs[i, j].set_ylabel('')
        # ====================================================================
        poc = pocs[2]
        dataset = 'CIFAR10'
        title = """CIFAR-10 ({})""".format(poc)
        i = 1
        j = 2
        self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
                             experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
                             hue='Strategy')
        legend = axs[i, j].get_legend()
        print("legenda: ", legend)
        # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        axs[i, j].get_legend().remove()
        axs[i, j].set_xlabel('')
        axs[i, j].set_ylabel('')
        # =========================///////////================================
        fig.suptitle("""Exp. {}""".format(str(experiment)), fontsize=16)
        plt.subplots_adjust(right=0.9)
        fig.legend(
                   loc="lower right")
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
    datasets = ['MNIST']
    clients = '50'
    model = 'DNN'
    file_type = 'evaluate_client.csv'

    joint_plot = JointAnalysis()
    joint_plot.build_filenames_and_read(strategies, pocs, datasets, experiments, clients, model, file_type)