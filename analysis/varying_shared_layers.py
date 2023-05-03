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
            parameters = float(df['Size of parameters'].mean())
            config = float(df['Size of config'].mean())
            acc = float(df['Accuracy'].mean())
            total_size = parameters + config

            return pd.DataFrame({'Size of parameters (bytes)': [parameters], 'Communication cost (bytes)': [total_size], 'Accuracy': [acc]})
        df = df[['Accuracy', 'Round', 'Size of parameters', 'Size of config', 'Strategy', 'Shared layers']].groupby(by=['Strategy', 'Shared layers', 'Round']).apply(lambda e: strategy(e)).reset_index()[['Accuracy', 'Size of parameters (bytes)', 'Communication cost (bytes)', 'Strategy', 'Shared layers', 'Round']]
        print(df)
        df['Accuracy (%)'] = df['Accuracy'] * 100
        df['Accuracy (%)'] = df['Accuracy (%)'].round(4)
        x_column = 'Round'
        y_column = 'Accuracy (%)'
        hue = 'Shared layers'
        title = """Alpha_{}""".format(alpha)
        base_dir = """analysis/output/torch/varying_shared_layers/alpha_{}/""".format(alpha)
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
                  type=1)
        x_column = 'Round'
        y_column = 'Communication cost (bytes)'
        hue = 'Shared layers'
        title = ""
        line_plot(df=df,
                  base_dir=base_dir,
                  file_name="evaluate_client_communication_cost_round_varying_shared_layers_lineplot",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue,
                  hue_order=layer_selection_evaluate,
                  type=1)

        filename = base_dir + "csv/comparison.csv"
        df.to_csv(filename, index=False)


if __name__ == '__main__':
    """
        This code generates a joint plot (multiples plots in one image) and a table of accuracy.
        It is done for each experiment.
    """

    strategy = "FedPredict"
    type = "torch"
    aggregation_method = "None"
    fraction_fit = 0.5
    num_clients = 10
    model_name = "CNN"
    dataset = "CIFAR10"
    alpha = 1.0
    num_rounds = 20
    epochs = 1
    layer_selection_evaluate = [1, 2, 3, 4]
    comment = ""

    Varying_Shared_layers(tp=type, strategy_name=strategy, fraction_fit=fraction_fit, aggregation_method=aggregation_method, new_clients=False, new_clients_train=False, num_clients=num_clients,
                          model_name=model_name, dataset=dataset, class_per_client=2, alpha=alpha, num_rounds=num_rounds, epochs=epochs,
                          comment=comment, layer_selection_evaluate=layer_selection_evaluate).start()
