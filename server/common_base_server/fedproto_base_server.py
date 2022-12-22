import flwr as fl
import numpy as np
import time
from pathlib import Path

from flwr.common import FitIns
from flwr.server.strategy.aggregate import weighted_loss_avg

from server.common_base_server.fedavg_base_server import FedAvgBaseServer
import shutil
import tensorflow as tf
import torch
import random
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
torch.manual_seed(0)
class FedProtoBaseServer(FedAvgBaseServer):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
                 strategy_name='FedProto',
                 model_name='',
                 new_clients=False):

        super().__init__(aggregation_method=aggregation_method,
                         n_classes=n_classes,
                         fraction_fit=fraction_fit,
                         num_clients=num_clients,
                         num_rounds=num_rounds,
                         decay=decay,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name='FedProto',
                         model_name=model_name,
                         new_clients=new_clients)

        self.global_protos = [np.array([np.nan]) for i in range(self.n_classes)]

        self.create_folder()

    def create_folder(self):

        directory = """fedproto_saved_weights/{}/""".format(self.model_name)
        if Path(directory).exists():
            shutil.rmtree(directory)
        for i in range(self.num_clients):
            Path("""fedproto_saved_weights/{}/{}/""".format(self.model_name, i)).mkdir(parents=True, exist_ok=True)

    def aggregate_fit(self, server_round, results, failures):
        weights_results = []

        # print("tamanho results: ", len(results))
        for _, fit_res in results:
            client_id = str(fit_res.metrics['cid'])
            protos_samples_per_class = fit_res.metrics['protos_samples_per_class']
            proto = fit_res.metrics['proto']

            if self.aggregation_method not in ['POC', 'FedLTA'] or int(server_round) <= 1:
                weights_results.append((fl.common.parameters_to_ndarrays(fit_res.parameters), protos_samples_per_class, proto))

            else:
                if client_id in self.selected_clients:
                    weights_results.append((fl.common.parameters_to_ndarrays(fit_res.parameters), protos_samples_per_class, proto))

        # print(f'LEN AGGREGATED PARAMETERS: {len(weights_results)}')
        # print("rodada: ", server_round)
        parameters_aggregated = fl.common.ndarrays_to_parameters(self._aggregate_proto(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated

    def _aggregate_proto(self, results):
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = {i: 0 for i in range(self.n_classes)}
        num_examples_total_clients = {i: 0 for i in range(self.n_classes)}
        for _, num_examples, p in results:

            for key in num_examples:

                num_examples_total[key] += num_examples[key]
                if num_examples[key] > 0:
                    num_examples_total_clients[key] += 1

        print("Quantidade por classe")
        print(num_examples_total)
        proto_shape = None
        # Create a list of weights, each multiplied by the related number of examples

        agg_protos_label = {i: None for i in range(self.n_classes)}
        for local_protos, num_examples, p in results:
            for label in range(self.n_classes):
                if agg_protos_label[label] is None:
                    agg_protos_label[label] = (local_protos[label]*num_examples[label])/num_examples_total[label]
                else:
                    # print("receber: ", agg_protos_label[label].shape, " ir: ", ((local_protos[label]*num_examples[label])/num_examples_total[label]).shape)
                    agg_protos_label[label] += (local_protos[label]*num_examples[label])/num_examples_total[label]
        # print("formato saida1: ", [agg_protos_label[i].shape for i in agg_protos_label])
        for key in agg_protos_label:
            proto = agg_protos_label[key]
            agg_protos_label[key] = [(np.array(proto) / num_examples_total_clients[key])]

        # for key in range(self.n_classes):
        #     for proto, num_examples in results:
        #
        #         # if key > len(proto) - 1:
        #         #     print("zero")
        #         #     continue
        #
        #         if num_examples[key] > 0:
        #             # print("umm", proto[key])
        #             if len(sum_protos[key]) == 0:
        #                 # print("umm", proto[key])
        #                 # print(sum_protos[key], proto[key], num_examples[key])
        #                 sum_protos[key] = proto[key]*num_examples[key]
        #
        #             else:
        #                 # print("dois")
        #                 # print("dois", sum_protos[key], proto[key][0], num_examples[key])
        #                 sum_protos[key] += proto[key]*num_examples[key]
        #
        #     if len(sum_protos[key]) > 0:
        #         # print("tres")
        #         sum_protos[key] = sum_protos[key]/(num_examples_total[key] * num_examples_total_clients[key])

        weighted_weights = [
            np.array(agg_protos_label[key]) for key in agg_protos_label
        ]
        tamanho = weighted_weights[0].shape
        # print("formato saida2: ", [i.shape for i in weighted_weights])
        weighted_weights = [i[0] for i in weighted_weights]
        # print("formato saida3: ", [i.shape for i in weighted_weights])
        for w in weighted_weights:
            if np.sum(w) == 0:
                print("zerado")
                exit()
            # if w.shape != tamanho:
            #     print("diferente: ", w.shape, tamanho)
            #     exit()
        # print("ponderado0")
        # print(weighted_weights)
        # Ensure that the server holds prototypes of all classes even that prototypes were not generated in the current round for a specific class
        for i in range(self.n_classes):
            new_proto = weighted_weights[i]
            if np.isnan(new_proto).any() and not np.isnan(self.global_protos[i]).any():
                weighted_weights[i] = self.global_protos[i]
        self.global_protos = weighted_weights
        # print("ponderado1")
        # print(weighted_weights)

        return weighted_weights

    def aggregate_evaluate(
            self,
            server_round,
            results,
            failures,
    ):

        local_list_clients = []
        self.list_of_clients = []
        self.list_of_accuracies = []
        accs = []

        for response in results:
            client_id = response[1].metrics['cid']
            client_accuracy = float(response[1].metrics['accuracy'])
            accs.append(client_accuracy)

            local_list_clients.append((client_id, client_accuracy))

        local_list_clients.sort(key=lambda x: x[1])

        self.list_of_clients = [str(client[0]) for client in local_list_clients]
        self.list_of_accuracies = [float(client[1]) for client in local_list_clients]

        accs.sort()
        self.average_accuracy = np.mean(accs)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # print("recebidas: ", [r.metrics["accuracy"] for _, r in results])

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        current_accuracy = accuracy_aggregated

        print(f"Round {server_round} accuracy aggregated from client results: {accuracy_aggregated}")

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate loss mse
        loss_mse_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.metrics['mse_loss'])
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        top5 = np.mean(accs[-5:])
        top1 = accs[-1]

        base = f"logs/{self.strategy_name}/new_clients_{self.new_clients}/{self.num_clients}/{self.model_name}/{self.dataset}/"
        filename_server = f"{base}server.csv"
        data = [time.time()-self.start_time, server_round, accuracy_aggregated, top5, top1]

        self._write_output(filename=filename_server,
                           data=data
                           )

        metrics_aggregated = {
            "loss_mse": loss_mse_aggregated,
            "accuracy": accuracy_aggregated,
            "top-3": top5,
            "top-1": top1
        }

        return loss_aggregated, metrics_aggregated