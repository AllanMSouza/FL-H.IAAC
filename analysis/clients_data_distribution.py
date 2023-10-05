import sys
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from base_plots import bar_plot, line_plot, ecdf_plot, box_plot, violin_plot, hist_plot, stacked_plot
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import seaborn as sns
import scipy.stats as st
import os
import pickle


class CLient:

    def __init__(self, cid, dataset_name, n_clients, class_per_client, alpha, classes=10):

        self.cid = cid
        self.filename_train = f"dataset_utils/data/{dataset_name}/{n_clients}_clients/classes_per_client_{class_per_client}/alpha_{alpha}/{cid}/idx_train_{cid}.pickle"
        self.filename_test = f"dataset_utils/data/{dataset_name}/{n_clients}_clients/classes_per_client_{class_per_client}/alpha_{alpha}/{cid}/idx_test_{cid}.pickle"
        self.alpha = alpha
        self.dataset = dataset_name
        self.classes = classes
        self.samples_per_class = {i: 0 for i in range(classes)}
        self.samples_per_class_percentage = {i: 0 for i in range(classes)}
        self.unique_classes = 0

    def start(self):


        try:
            transform_train = transforms.Compose(
                [transforms.Resize((32, 32)),  # resises the image so it can be perfect for our model.
                 transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                 transforms.RandomRotation(10),  # Rotates the image to a specified angel
                 transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                 # Performs actions like zooms, change shear angles.
                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
                 transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize all the images
                 ])

            transform_test = transforms.Compose([transforms.Resize((32, 32)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ])

            # transform = transforms.Compose(
            #     [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
            # transform_train = transform
            # transform_test = transform

            if self.dataset == "CIFAR10":
                dir_path = "dataset_utils/data/CIFAR10/raw_data/"
                training_dataset = datasets.CIFAR10(root=dir_path, train=True, download=False,
                                                    transform=transform_train)  # Data augmentation is only done on training images
                validation_dataset = datasets.CIFAR10(root=dir_path, train=False, download=False,
                                                      transform=transform_train)
            elif self.dataset == "EMNIST":
                dir_path = "dataset_utils/data/EMNIST/raw_data/"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                # Get EMNIST data
                transform = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(10),
                                                transforms.Normalize([0.5], [0.5])])

                training_dataset = datasets.EMNIST(
                    root=dir_path, train=True, download=False, transform=transform, split='balanced')
                validation_dataset = datasets.EMNIST(
                    root=dir_path, train=False, download=False, transform=transform, split='balanced')

            with open(self.filename_train, 'rb') as handle:
                idx_train = pickle.load(handle)

            with open(self.filename_test, 'rb') as handle:
                idx_test = pickle.load(handle)

            y = training_dataset.targets
            y = np.concatenate((y, validation_dataset.targets))
            y_train = y[idx_train]
            y_test = y[idx_test]
            y = np.concatenate((y_train, y_test))
            total = len(y)
            for i in y:
                self.samples_per_class[int(i)] += 1

            self.unique_classes = len(pd.Series(y).unique().tolist())


            for i in range(self.classes):

                self.samples_per_class_percentage[i] = (self.samples_per_class[i]/total) * 100

            client =  []
            samples = []
            classs = []
            dataset = []
            alpha = []

            for clas in self.samples_per_class_percentage:

                client.append(self.cid)
                samples.append(self.samples_per_class_percentage[clas])
                classs.append(clas)
                dataset.append(self.dataset)
                alpha.append(self.alpha)

            return client, samples, classs, dataset, alpha, self.unique_classes


        except Exception as e:
            print("Select CIFAR10")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)



class Varying_Shared_layers:

    def __init__(self, tp, num_clients, dataset,
                 class_per_client, alpha):

        self.type = tp
        self.num_clients = num_clients
        self.dataset = dataset
        self.class_per_client = class_per_client
        self.alpha = alpha
        self.clients = []
        self.base_dir = """analysis/output/torch/dataset_{}/alpha_{}/""".format(str(self.dataset), str(self.alpha))

        for alpha in self.alpha:
            for dataset in self.dataset:
                classes = {'CIFAR10': 10, 'EMNIST': 47}[dataset]
                for i in range(self.num_clients):
                    self.clients.append(CLient(i, dataset_name=dataset, n_clients=num_clients, class_per_client=class_per_client, alpha=alpha, classes=classes))

    def start(self):

        # client analysis
        clients = []
        percentages = []
        classes = []
        datasets = []
        alphas = []

        # summary
        alpha_summary = {dataset: {i: [] for i in self.alpha} for dataset in self.dataset}
        unique_classes_summary = []
        alpha_unique_classes_summary_total_clients = {}

        for client in self.clients:
            client, percentage, classs, dataset, alpha, uc = client.start()
            clients += client
            percentages += percentage
            classes += classs
            datasets += dataset
            alphas += alpha
            alpha_summary[dataset[0]][alpha[0]].append(uc)

        for dataset in self.dataset:
            alphas_list = []
            classes_list = []
            count = []
            media = {alpha: [] for alpha in self.alpha}
            for alpha in self.alpha:
                unique_classes = alpha_summary[dataset][alpha]
                classes = {'CIFAR10': 10, 'EMNIST': 47}[dataset]
                unique_classes_count = {i: 0 for i in range(1, classes + 1)}
                for unique in unique_classes:
                    unique_classes_count[unique] += 1

                media[alpha].append(uc)

                alphas_list += [alpha] * len(list(unique_classes_count.keys()))
                classes_list += list(unique_classes_count.keys())
                count += list(unique_classes_count.values())

            print("Dataset: ", dataset)
            for alpha in self.alpha:

                print("Media: ", np.mean(media[alpha]))



        # df = pd.DataFrame({'Client': clients, 'Samples (%)': percentages, 'Class': classes, 'Dataset': datasets, '\u03B1': alphas})
        # print(df)
        # print("Unico: ", unique_classes_count)
        # self.df = df
        # self.dataset_alpha_analysis()

            # new_alp = []
            # new_clas = []
            # new_coun = []
            # intervals = {'CIFAR10': [[3, 5, 8, 10], ['[1-3)', '[3-5)', '[5-8)', '[8-10]']], 'EMNIST': [[10, 20, 30, 40, 47]]}
            # for alp, clas, coun in zip(alphas_list, classes_list, count):


            self.df_summary = pd.DataFrame({'\u03B1': alphas_list, 'Unique_classes': classes_list, 'Total of clients': count})
            self.df_summary['Total_of_clients_(%)'] = (self.df_summary['Total of clients']/40)*100
            print("sumario antes: ")
            print(self.df_summary)


            # self.df_summary = self.df_summary.groupby('\u03B1').mean().reset_index()
            # print("sumario")
            # print(self.df_summary)
            stacked_plot(df=self.df_summary, base_dir=self.base_dir, file_name="""unique_classes_{}""".format(dataset), x_column='\u03B1', y_column='Total_of_clients_(%)', title="""{}""".format(dataset), hue='Unique_classes')

            bar_plot(df=self.df_summary, base_dir=self.base_dir, file_name="""unique_classes_{}""".format(dataset), x_column='\u03B1', y_column='Total_of_clients_(%)', hue='Unique_classes', title="""{}""".format(dataset))

    # def summary_alphas_clients_unique_classes(self):
    #
    #     self.un

    # def rotate_df(self, df):




    def plot(self, ax, df, dataset, alpha, x_column, y_column, base_dir, hue, i, j, y_max=100):

        df = df.query("""Dataset == '{}' and \u03B1 == {}""".format(self.dataset[i],
                                                                       self.alpha[j]))

        # print("filtro: ", """Dataset == '{}' and \u03B1 == {}""".format(self.dataset[i],
        #                                                                self.alpha[j]))
        # print("filtrado: ", df)
        title = """\u03B1={}""".format(self.alpha[j])
        bar_plot(
            ax=ax[j],
            df=df,
            base_dir=base_dir,
            file_name="",
            x_column=x_column,
            y_column=y_column,
            title=title,
            hue=hue,
            y_lim=True,
            y_max=y_max,
            y_min=0)
        ax[j].get_legend().remove()
        ax[j].set_xlabel('')
        ax[j].set_ylabel('')
        # ax[i, j].get_legend().remove()
        # ax[i, j].set_xlabel('')
        # ax[i, j].set_ylabel('')

    def dataset_alpha_analysis(self):

        print("fora")
        if len(self.dataset) >= 1 and len(self.alpha) >= 1:
            print("entrou")
            fig, ax = plt.subplots(len(self.dataset), len(self.alpha), sharex='all', sharey='all', figsize=(6, 6))
            y_max = 1.4
            x_column = 'Client'
            y_column = 'Samples (%)'
            hue = 'Class'

            for i in range(len(self.dataset)):
                dataset = self.dataset[i]
                for j in range(len(self.alpha)):
                    a = self.alpha[j]
                    # df = self.df.query("""\u03B1 == {} and Dataset == '{}'""".format(a, dataset))
                    self.plot(ax, self.df, dataset, a, x_column, y_column, self.base_dir, hue, i, j, y_max=100)


            fig.suptitle("", fontsize=16)
            fig.supxlabel(x_column, y=-0.02)
            fig.supylabel(y_column, x=-0.005)
            # plt.tight_layout(pad=0.5)

            plt.subplots_adjust(wspace=0.07, hspace=0.14)
            lines_labels = [ax[0].get_legend_handles_labels()]
            # lines_labels = [ax[0, 0].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines, labels, title="""\u03B1={}""".format(alpha), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.06))
            figure = fig.get_figure()
            Path(self.base_dir + "png/").mkdir(parents=True, exist_ok=True)
            Path(self.base_dir + "svg/").mkdir(parents=True, exist_ok=True)
            filename = """datasets_alphas""".format(str(self.dataset), str(self.alpha))
            figure.savefig(self.base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
            figure.savefig(self.base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)

if __name__ == '__main__':
    """
        This code generates a joint plot (multiples plots in one image) and a table of accuracy.
        It is done for each experiment.
    """

    strategy = "FedPredict"
    type_model = "torch"
    aggregation_method = "None"
    fraction_fit = 0.3
    num_clients = 40
    dataset = ["CIFAR10", "EMNIST"]
    alpha = [0.1, 1.0, 3.0, 5.0]
    num_rounds = 50

    Varying_Shared_layers(tp=type_model, num_clients=num_clients,  dataset=dataset, class_per_client=2, alpha=alpha).start()


