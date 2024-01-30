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
import numpy as np

from scipy.stats import entropy
import os
import pickle


class CLient:

    def __init__(self, cid, dataset_name, n_clients, class_per_client, alpha, classes=10):

        self.cid = cid
        self.filename_train = f"dataset_utils/data/{dataset_name.replace('CIFAR-10', 'CIFAR10')}/{n_clients}_clients/classes_per_client_{class_per_client}/alpha_{alpha}/{cid}/idx_train_{cid}.pickle"
        self.filename_test = f"dataset_utils/data/{dataset_name.replace('CIFAR-10', 'CIFAR10')}/{n_clients}_clients/classes_per_client_{class_per_client}/alpha_{alpha}/{cid}/idx_test_{cid}.pickle"
        self.alpha = alpha
        self.dataset = dataset_name.replace("CIFAR10", "CIFAR-10")
        self.classes = classes
        self.samples_per_class = {}
        self.samples_per_class_percentage = {}
        self.unique_classes = 0
        self.min_samples_per_class = 100
        self.imbalance_level = 0
        self.n_clients = n_clients

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

            if self.dataset == "CIFAR-10":
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

            elif self.dataset == "GTSRB":

                dir_path = "dataset_utils/data/GTSRB/raw_data/"
                # antigo

                training_dataset, validation_dataset = self.load_data_gtsrb(dir_path)
                dataset_image = []
                dataset_label = []
                for i in range(3):
                    dataset_image.extend(training_dataset.samples)
                    dataset_label.extend(training_dataset.targets)
                dataset_image = np.array(dataset_image)
                y = np.array(dataset_label)
                validation_dataset = copy.deepcopy(training_dataset)
                validation_dataset.targets = np.array([])

            elif self.dataset in ["Cologne", "WISDM-WATCH", "WISDM-P"]:

                filename_train = self.filename_train.replace("pickle", "csv")
                filename_test = self.filename_test.replace("pickle", "csv")

                train = pd.read_csv(filename_train)
                test = pd.read_csv(filename_test)
                y_train = np.array([i for i in train['Y'].to_numpy().astype(np.int32)])
                y_test = np.array([i for i in test['Y'].to_numpy().astype(np.int32)])

                y = np.concatenate((y_train, y_test))
                total = len(y)

            if self.dataset not in ["Cologne", "WISDM-WATCH", "WISDM-P"]:

                with open(self.filename_train, 'rb') as handle:
                    idx_train = pickle.load(handle)

                with open(self.filename_test, 'rb') as handle:
                    idx_test = pickle.load(handle)

                if self.dataset != 'GTSRB':
                    y = training_dataset.targets
                    y = np.concatenate((y, validation_dataset.targets))
                y_train = y[idx_train]
                y_test = y[idx_test]
                y = np.concatenate((y_train, y_test))
                total = len(y)


            self.unique_classes = len(pd.Series(y).unique().tolist())
            self.unique_classes_list = pd.Series(y).unique().tolist()
            self.samples_per_class = {i: 0 for i in self.unique_classes_list}
            for i in y:
                self.samples_per_class[int(i)] += 1

            self.imbalance_level = 0
            # self.min_samples_per_class = total/self.classes
            self.min_samples_per_class = int(len(y)/3/len(self.unique_classes_list))
            for class_ in self.samples_per_class:
                if self.samples_per_class[class_] < self.min_samples_per_class:
                    self.imbalance_level += 1

            self.imbalance_level = self.imbalance_level * 100/ len(self.samples_per_class)




            for i in self.unique_classes_list:

                self.samples_per_class_percentage[i] = (self.samples_per_class[i]/total) * 100

            percentages = list(self.samples_per_class_percentage.values())
            base = 2  # work in units of bits

            H = entropy(percentages, base=base)
            balance = H/np.log2(self.unique_classes)


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

            # print(client, samples, classs, dataset, alpha, [self.unique_classes], [self.imbalance_level], [self.dataset])
            return client, samples, classs, dataset, alpha, [self.unique_classes], balance, self.dataset, self.imbalance_level


        except Exception as e:
            print("Select CIFAR10")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def load_data_gtsrb(self, data_path):
        """Load ImageNet (training and val set)."""

        try:
            # Load ImageNet and normalize
            traindir = os.path.join(data_path, "Train")

            trainset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [   transforms.Resize((32, 32)),
                        # transforms.RandomHorizontalFlip(),
                        # transforms.RandomResizedCrop(224),
                        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                        # transforms.RandomRotation(degrees=60, expand=False),
                        transforms.ToTensor(),

                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )
            )
            # print("tee")
            # print(type(trainset.classes), trainset.classes, len(trainset.classes))
            # print(type(trainset.class_to_idx), trainset.class_to_idx)
            # print()

            return trainset, None

        except Exception as e:
            print("load data gtrsb")
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
                classes = {'CIFAR-10': 10, 'EMNIST': 47, 'GTSRB': 43, 'Cologne': 12, 'WISDM-WATCH': 12, 'WISDM-P': 12}[dataset]
                for i in range(self.num_clients[dataset]):
                    self.clients.append(CLient(i, dataset_name=dataset, n_clients=self.num_clients[dataset], class_per_client=class_per_client, alpha=alpha, classes=classes))

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
        unique_classes_list = []
        dt_list = []
        alpha_list = []
        balance_level_list = []
        imbalance_level_list = []

        for client in self.clients:
            client, percentage, classs, dataset, alpha, uc, balance_level, dt, imbalance_level = client.start()
            clients += client
            percentages += percentage
            classes += classs
            datasets += dataset
            dt_list.append(dt)
            alphas += alpha
            alpha_summary[dataset[0]][alpha[0]].append(uc)
            clas = {'CIFAR-10': 10, 'EMNIST': 47, 'GTSRB': 43, 'Cologne': 12, 'WISDM-WATCH': 12, 'WISDM-P': 12}[dt]
            unique_classes_list += [uc[0]*100/clas]
            alpha_list.append(alpha[0])
            balance_level_list.append(balance_level)
            imbalance_level_list.append(imbalance_level)

        # for dataset in self.dataset:
        #     alphas_list = []
        #     classes_list = []
        #     count = []
        #     media = {alpha: [] for alpha in self.alpha}
        #     for alpha in self.alpha:
        #         unique_classes = alpha_summary[dataset][alpha]
        #         classes = {'CIFAR10': 10, 'EMNIST': 47, 'GTSRB': 43}[dataset]
        #         unique_classes_count = {i: 0 for i in range(1, classes + 1)}
        #         for unique in unique_classes:
        #             unique_classes_count[unique] += 1
        #
        #         media[alpha].append(uc)
        #
        #         alphas_list += [alpha] * len(list(unique_classes_count.keys()))
        #         classes_list += list(unique_classes_count.keys())
        #         count += list(unique_classes_count.values())
        #
        #     print("Dataset: ", dataset)
        #     for alpha in self.alpha:
        #
        #         print("Media: ", np.mean(media[alpha]))



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


            # self.df_summary = pd.DataFrame({'\u03B1': alphas_list, 'Unique_classes': classes_list, 'Total of clients': count})
            # self.df_summary['Total_of_clients_(%)'] = (self.df_summary['Total of clients']/self.num_clients[dataset])*100
            # print("sumario antes: ")
            # print(self.df_summary)

        dt_list = [i.replace("WISDM-WATCH", "WISDM-W") for i in dt_list]

        df_unique_classes = pd.DataFrame({'\u03B1': alpha_list, 'Dataset': dt_list, 'Classes (%)': unique_classes_list, 'Balance level': balance_level_list, 'Imbalance level (%)': imbalance_level_list})


            # self.df_summary = self.df_summary.groupby('\u03B1').mean().reset_index()
            # print("sumario")
            # print(self.df_summary)
            # stacked_plot(df=df_unique_classes, base_dir=self.base_dir, file_name="""unique_classes_{}""".format(dataset), x_column='\u03B1', y_column='Classes (%)', title="""{}""".format(dataset), hue='Dataset')
        print(df_unique_classes)
        print(self.base_dir)

        self.plot_(df_unique_classes)
        self.plot_(df_unique_classes)

    def plot_(self, df_unique_classes):
        x_order = [0.1, 0.5, 1.0]

        # hue_order = ['EMNIST', 'CIFAR-10', 'GTSRB', 'WISDM-WATCH', 'WISDM-P']
        hue_order = ['Cologne', 'WISDM-W', 'WISDM-P']

        fig, axs = plt.subplots(2, 1, sharex='all', sharey='all', figsize=(6, 6))
        bar_plot(df=df_unique_classes, base_dir=self.base_dir, ax=axs[0], x_order=x_order, file_name="""unique_classes_{}""".format(self.dataset), x_column='\u03B1', y_column='Classes (%)', title="""Clients' local classes""", tipo="classes", y_max=100, hue='Dataset', hue_order=hue_order)
        i = 0
        axs[i].get_legend().remove()
        axs[i].legend(fontsize=10)
        axs[i].set_xlabel('')
        # axs[i].set_ylabel('')
        # bar_plot(df=df_unique_classes, base_dir=self.base_dir, file_name="""balance_level_{}""".format(self.dataset),
        #          x_column='\u03B1', y_column='Balance level', title="""""", y_max=1, hue='Dataset', tipo="balance")
        bar_plot(df=df_unique_classes, base_dir=self.base_dir, ax=axs[1], x_order=x_order, file_name="""imbalance_level_{}""".format(self.dataset),
                 x_column='\u03B1', y_column='Imbalance level (%)', title="""Dataset imbalance level""", y_max=100, hue='Dataset', tipo="balance", hue_order=hue_order)
        i = 1
        axs[i].get_legend().remove()
        # axs[i].legend(fontsize=7)
        # axs[i].set_xlabel('')
        # axs[i].set_ylabel('')
        fig.suptitle("", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.07, hspace=0.14)
        fig.savefig(
            """{}unique_imbalance_level_{}_clients.png""".format(self.base_dir,
                                                                            list(self.num_clients.keys())), bbox_inches='tight', dpi=400)
        fig.savefig(
            """{}unique_imbalance_level_{}_clients.svg""".format(self.base_dir,
                                                                            list(self.num_clients.keys())), bbox_inches='tight', dpi=400)



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
            filename = """datasets_alphas""".format(str(list(self.dataset.keys())), str(self.alpha))
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
    num_clients = {'GTSRB': 20, 'EMNIST': 20, 'CIFAR-10': 20, 'Cologne': 20, 'WISDM-WATCH': 20, 'WISDM-P': 20}
    dataset = ["Cologne", "WISDM-WATCH", "WISDM-P"]
    alpha = [0.1, 0.5, 1.0]
    num_rounds = 50

    Varying_Shared_layers(tp=type_model, num_clients=num_clients,  dataset=dataset, class_per_client=2, alpha=alpha).start()


