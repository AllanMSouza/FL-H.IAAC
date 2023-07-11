import numpy as np
import pandas as pd
from base_plots import bar_plot, line_plot, ecdf_plot
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

    def start(self):

        self.build_filenames()
        self.evaluate_client_analysis_shared_layers(self.layer_selection_evaluate)

    def build_filenames(self):

        file = "evaluate_client.csv"
        df_concat = None
        for layers in self.layer_selection_evaluate:
            filename = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/FL-H.IAAC/" + f"logs/{self.type}/{self.strategy_name}-{self.aggregation_method}-{self.fraction_fit}/new_clients_{self.new_clients}_train_{self.new_clients_train}/{self.num_clients}/{self.model_name}/{self.dataset}/classes_per_client_{self.class_per_client}/alpha_{self.alpha}/{self.num_rounds}_rounds/{self.epochs}_local_epochs/{self.comment}_comment/{str(layers)}_layer_selection_evaluate/{file}"
            df = pd.read_csv(filename)
            df['Shared layers'] = np.array([layers] * len(df))
            df['Strategy'] = np.array([self.strategy_name] * len(df))
            if df_concat is None:
                df_concat = df
            else:
                df_concat = pd.concat([df_concat, df], ignore_index=True)

        self.df_concat = df_concat
        print(df_concat)



    def evaluate_client_analysis_shared_layers(self, layer_selection_evaluate):
        # acc
        df = self.df_concat
        def strategy(df):
            parameters = float(df['Size of parameters'].mean())/1000000
            config = float(df['Size of config'].mean())/1000000
            acc = float(df['Accuracy'].mean())
            acc_gain_per_byte = acc/parameters
            total_size = parameters + config

            return pd.DataFrame({'Size of parameters (MB)': [parameters], 'Communication cost (MB)': [total_size], 'Accuracy': [acc], 'Accuracy gain per MB': [acc_gain_per_byte]})
        df = df[['Accuracy', 'Round', 'Size of parameters', 'Size of config', 'Strategy', 'Shared layers']].groupby(by=['Strategy', 'Shared layers', 'Round']).apply(lambda e: strategy(e)).reset_index()[['Accuracy', 'Size of parameters (MB)', 'Communication cost (MB)', 'Strategy', 'Shared layers', 'Round', 'Accuracy gain per MB']]
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
                shared_layers_list[i] = "FedPredict-v2"
                sort[shared_layer] = shared_layers_list[i]
                continue
            new_shared_layer = "{"
            for layer in shared_layer:
                if len(new_shared_layer) == 1:
                    new_shared_layer += layer
                else:
                    new_shared_layer += ", " + layer
            new_shared_layer += "}"

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

        title = """Alpha={}; Layer order={}""".format(alpha, comment)
        base_dir = """analysis/output/torch/varying_shared_layers/{}/{}_clients/{}_fraction_fit/alpha_{}/{}_comment/""".format(self.dataset, self.num_clients, self.fraction_fit, alpha, self.comment)
        os.makedirs(base_dir + "png/", exist_ok=True)
        os.makedirs(base_dir + "svg/", exist_ok=True)
        os.makedirs(base_dir + "csv/", exist_ok=True)
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_acc_round_varying_shared_layers_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  hue_order=layer_selection_evaluate,
                  type=1,
                  y_lim=True,
                  y_min=10,
                  y_max=80)
        x_column = 'Round'
        y_column = 'Communication cost (MB)'
        hue = 'Shared layers'
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_communication_cost_round_varying_shared_layers_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  hue_order=layer_selection_evaluate,
                  type=1,
                  y_lim=True,
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
                  file_name="evaluate_client_accuracy_gain_per_MB_varying_shared_layers_lineplot",
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

            round = df['Round'].tolist()[0]
            df_aux = df_aux[df_aux['Round'] == round]
            target = df_aux[df_aux['Shared layers'] == "{1, 2, 3, 4}"]
            target_acc = target['Accuracy (%)'].tolist()[0]
            target_size = target['Communication cost (MB)'].tolist()[0]
            acc = df['Accuracy (%)'].tolist()[0]
            size = df['Communication cost (MB)'].tolist()[0]
            acc_reduction = target_acc - acc
            size_reduction = (target_size - size)
            # acc_weight = 1
            # size_weight = 1
            # acc_score = acc_score *acc_weight
            # size_reduction = size_reduction * size_weight
            # score = 2*(acc_score * size_reduction)/(acc_score + size_reduction)
            # if df['Shared layers'].tolist()[0] == "{1, 2, 3, 4}":
            #     acc_reduction = 0.0001
            #     size_reduction = 0.0001

            return pd.DataFrame({'Accuracy reduction (%)': [acc_reduction], 'Communication reduction (MB)': [size_reduction]})

        df = df[['Accuracy (%)', 'Size of parameters (MB)', 'Communication cost (MB)', 'Strategy', 'Shared layers',
             'Round', 'Accuracy gain per MB']].groupby(
            by=['Strategy', 'Round', 'Shared layers']).apply(lambda e: comparison_with_shared_layers(e, df)).reset_index()[
            ['Strategy', 'Round', 'Shared layers', 'Accuracy reduction (%)', 'Communication reduction (MB)']]

        print("Final: ", df)
        df = df[df['Shared layers'] != "{1, 2, 3, 4}"]
        layer_selection_evaluate =  ['FedPredict-v2', '{1}']
        print("menor: ", df['Accuracy reduction (%)'].min())
        print("Fed", df[df['Shared layers'] == 'FedPredict-v2'][['Accuracy reduction (%)', 'Round']])

        x_column = 'Round'
        y_column = 'Accuracy reduction (%)'
        hue = 'Shared layers'
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_accuracy_reduction_varying_shared_layers_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  hue_order=layer_selection_evaluate,
                  type=1,
                  log_scale=False,
                  y_lim=True,
                  y_max=10,
                  y_min=-3)

        x_column = 'Round'
        y_column = 'Communication reduction (MB)'
        hue = 'Shared layers'
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_communication_reduction_varying_shared_layers_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  hue_order=layer_selection_evaluate,
                  type=1,
                  log_scale=True,
                  y_lim=True,
                  y_max=4,
                  y_min=0,
                  n=1)


if __name__ == '__main__':
    """
        This code generates a joint plot (multiples plots in one image) and a table of accuracy.
        It is done for each experiment.
    """

    strategy = "FedPredict"
    type_model = "torch"
    aggregation_method = "None"
    fraction_fit = 0.4
    num_clients = 20
    model_name = "CNN"
    dataset = "CIFAR10"
    alpha = float(2)
    num_rounds = 20
    epochs = 1
    # layer_selection_evaluate = [-1, 1, 2, 3, 4, 12, 13, 14, 123, 124, 134, 23, 24, 1234, 34]
    #layer_selection_evaluate = [1, 12, 123, 1234]
    # layer_selection_evaluate = [4, 34, 234, 1234]
    layer_selection_evaluate = [-1, 1234, 1]
    comment = "set"

    Varying_Shared_layers(tp=type_model, strategy_name=strategy, fraction_fit=fraction_fit, aggregation_method=aggregation_method, new_clients=False, new_clients_train=False, num_clients=num_clients,
                          model_name=model_name, dataset=dataset, class_per_client=2, alpha=alpha, num_rounds=num_rounds, epochs=epochs,
                          comment=comment, layer_selection_evaluate=layer_selection_evaluate).start()
