import csv
import json
import os
import pickle
from typing import Optional, List, Tuple, Union, Dict

import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, FitIns, FitRes, Scalar, parameters_to_ndarrays, EvaluateIns, \
    EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

# from battery import get_energy_by_completion_time, idle_power_deduction
# from model_builder import get_model


class Poc:
    def __init__(self, dataset_name: str, num_clients: int, num_classes: int, fit_fraction: float, eval_fraction: float,
                 min_fit: int,
                 min_eval: int, min_avail: int, learning_rate: float, results_dir: str, sim_id: str,
                 transmission_threshold: float, devices_profile: str, network_profiles: str, sim_idx: int, input_shape,
                 samples_per_client: list, d_temp_set_size: int):
        super().__init__()
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.fit_fraction = fit_fraction
        self.eval_fraction = eval_fraction
        self.min_fit = min_fit
        self.min_eval = min_eval
        self.min_avail = min_avail
        self.learning_rate = learning_rate
        self.results_dir = results_dir
        self.sim_id = sim_id
        self.transmission_threshold = transmission_threshold
        self.devices_profile = devices_profile
        self.network_profiles = network_profiles
        self.sim_idx = sim_idx
        self.input_shape = input_shape
        self.samples_per_client = samples_per_client
        self.d_temp_set_size = d_temp_set_size
        self.prob_per_client = [samples / sum(samples_per_client) for samples in samples_per_client]
        self.clients_info = {}
        self.last_loss = []
        self.net = None
        self.d_temp_set_cids = None
        self.round = 0

    def __repr__(self) -> str:
        return "power_of_choice"

    # def initialize_parameters(
    #         self, client_manager: ClientManager
    # ) -> Optional[Parameters]:
    #     """Initialize global model parameters."""
    #     self.net = get_model(self.dataset_name, self.input_shape, self.num_classes)
    #     network_profiles = None
    #
    #     with open(self.network_profiles, 'rb') as file:
    #         network_profiles = pickle.load(file)
    #
    #     with open(self.devices_profile, 'r') as file:
    #         json_dict = json.load(file)
    #         for key in json_dict.keys():
    #             self.clients_info[int(key)] = json_dict[key]
    #             self.clients_info[int(key)]['perc_budget_10'] = False
    #             self.clients_info[int(key)]['perc_budget_20'] = False
    #             self.clients_info[int(key)]['perc_budget_30'] = False
    #             self.clients_info[int(key)]['perc_budget_40'] = False
    #             self.clients_info[int(key)]['perc_budget_50'] = False
    #             self.clients_info[int(key)]['perc_budget_60'] = False
    #             self.clients_info[int(key)]['perc_budget_70'] = False
    #             self.clients_info[int(key)]['perc_budget_80'] = False
    #             self.clients_info[int(key)]['perc_budget_90'] = False
    #             self.clients_info[int(key)]['perc_budget_100'] = False
    #             self.clients_info[int(key)]['initial_battery'] = self.clients_info[int(key)]['battery']
    #             self.clients_info[int(key)]['network_profile'] = network_profiles[int(key)]
    #
    #     for cid in range(self.num_clients):
    #         self.last_loss.append((np.inf, cid))
    #
    #     client_manager.wait_for(self.num_clients)
    #
    #     return ndarrays_to_parameters(self.net.get_weights())

    def poc_configure_fit(self, server_round: int):

        # num_available = 0
        # for cid in self.clients_info:
        #     if self.clients_info[cid]['battery'] > 0:
        #         num_available += 1
        #
        # sample_size, min_num_clients = self.num_fit_clients(
        #     num_available
        # )

        if server_round % 2 != 0:
            selected_cids = self.sample_temp_set(self.d_temp_set_size)
            # self.decrease_battery(selected_cids)
            self.d_temp_set_cids = selected_cids
        else:
            self.round += 1
            selected_cids = self.sample_fit(sample_size, self.d_temp_set_cids)

            if len(selected_cids) > 0:
                selected_cids = self.filter_clients_to_train_by_predicted_behavior(selected_cids)

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of strategy."""
        num_clients = int(num_available_clients * self.fit_fraction)
        return max(num_clients, self.min_fit), self.min_avail

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available strategy for evaluation."""
        num_clients = int(num_available_clients * self.eval_fraction)
        return max(num_clients, self.min_eval), self.min_avail

    def weighted_matthews_ceof_avg(self, results: List[Tuple[int, float]]) -> float:
        """Aggregate evaluation results obtained from multiple strategy."""
        num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
        weighted_mcc = [num_examples * mcc for num_examples, mcc in results]
        return sum(weighted_mcc) / num_total_evaluation_examples

    def sample_temp_set(self, d_temp_set_size):
        sample_space = [cid for cid in self.clients_info if self.clients_info[cid]['battery'] > 0]

        temp_clients_prob = [self.prob_per_client[cid] for cid in sample_space]
        temp_clients_prob_sum = sum(temp_clients_prob)

        real_temp_size = min(d_temp_set_size, len(sample_space))

        temp_selected_idxs = np.random.choice(sample_space, real_temp_size, False,
                                              [prob / temp_clients_prob_sum for prob in temp_clients_prob])

        return temp_selected_idxs.tolist()

    # def sample_fit(self, num_clients, d_temp_set_cids):
    #
    #     clients_available = [self.last_loss[cid] for cid in d_temp_set_cids if self.clients_info[cid]['battery'] > 0]
    #
    #     if len(clients_available) == 0:
    #         return []
    #     clients_available.sort(key=lambda x: x[0], reverse=True)
    #
    #     rep = list(zip(*clients_available))
    #     selected_cids = rep[1][:num_clients]
    #
    #     return selected_cids

    def sample_eval(self, client_manager: ClientManager, num_clients):
        return client_manager.sample(num_clients)

    def filter_clients_to_train_by_predicted_behavior(self, selected_cids):
        # 1 - Atualiza tempo máximo de processamento
        # 2 - Atualiza bateria consumida pelos clientes em treinamento
        # 3 - Atualiza métricas: energia consumida, desperdício de energia, clientes dreanados, latência total
        # 4 - Identificar clientes que falharão a transmissão devido a alguma instabilidade
        # 5 - Atualizar métrica de consumo desperdiçado por falha do cliente
        # 6 - Atualiza lista de clientes que não completaram o treino por falta de bateria ou instabilidade da rede
        total_latency_round = 0.0
        total_energy_consumed = 0.0
        total_wasted_energy = 0.0
        round_depleted_battery_by_train = 0
        round_depleted_battery_total = 0
        round_transpassed_min_battery_level = 0
        max_latency = 0.0
        filtered_by_transmisssion = 0
        clients_to_not_train = []

        for cid in selected_cids:
            comp_latency = self.clients_info[cid]['train_latency']
            comm_latency = self.clients_info[cid]['comm_latency']
            avg_joules = self.clients_info[cid]["avg_joules"]

            client_latency = self.clients_info[cid]["total_train_latency"]
            client_consumed_energy = get_energy_by_completion_time(comp_latency, comm_latency, avg_joules)
            new_battery_value = self.clients_info[cid]['battery'] - client_consumed_energy

            if self.clients_info[cid]['battery'] >= self.clients_info[cid]['min_battery'] and new_battery_value < \
                    self.clients_info[cid]['min_battery']:
                round_transpassed_min_battery_level += 1

            if new_battery_value < 0:
                total_energy_consumed += self.clients_info[cid]['battery']
                total_wasted_energy += self.clients_info[cid]['battery']
                self.clients_info[cid]['battery'] = 0
                round_depleted_battery_by_train += 1
                round_depleted_battery_total += 1
                clients_to_not_train.append(cid)
            else:
                total_energy_consumed += client_consumed_energy
                self.clients_info[cid]['battery'] = new_battery_value

                if self.clients_info[cid]['network_profile'][self.round - 1] < self.transmission_threshold:
                    clients_to_not_train.append(cid)
                    total_wasted_energy += client_consumed_energy
                    filtered_by_transmisssion += 1
                else:
                    total_latency_round += client_latency

            if client_latency > max_latency and cid not in clients_to_not_train:
                max_latency = client_latency
        # 7 - Remove de clientes selecionados os que foram drenados pelo treinamento
        filtered_selected_cids = list(set(selected_cids).difference(clients_to_not_train))
        # 8 - Calcular consumo em estado de espera
        # 9 - Atualizar bateria de cada cliente
        # 10 - Atualizar clientes que foram drenados sem que seja pelo treino
        for cid in self.clients_info:
            old_battery_level = self.clients_info[cid]['battery']

            if old_battery_level > 0:
                if cid not in filtered_selected_cids:
                    new_battery_level = idle_power_deduction(old_battery_level, max_latency)
                else:
                    idle_time = max_latency - (self.clients_info[cid]['total_train_latency'])
                    new_battery_level = idle_power_deduction(old_battery_level, idle_time)

                if self.clients_info[cid]['battery'] >= self.clients_info[cid]['min_battery'] and new_battery_level < \
                        self.clients_info[cid]['min_battery']:
                    round_transpassed_min_battery_level += 1

                if new_battery_level <= 0:
                    self.clients_info[cid]['battery'] = 0
                    round_depleted_battery_total += 1
                else:
                    self.clients_info[cid]['battery'] = new_battery_level

        perc_budget_10, perc_budget_100, perc_budget_20, perc_budget_30, perc_budget_40, perc_budget_50, \
            perc_budget_60, perc_budget_70, perc_budget_80, perc_budget_90, avg_battery_perc, batteries_perc = \
            self.transpassed_budget()

        filename = self.results_dir + self.__repr__() + "_" + self.sim_id + "_" + str(
            self.sim_idx) + f"_system_metrics_frac_{self.fit_fraction}.csv"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'a') as log:
            log.write(f"{self.round},{total_latency_round},{total_energy_consumed},{total_wasted_energy},"
                      f"{len(selected_cids)},{round_depleted_battery_by_train},{round_depleted_battery_total},"
                      f"{filtered_by_transmisssion},{len(filtered_selected_cids)},"
                      f"{round_transpassed_min_battery_level},{perc_budget_10},{perc_budget_20},{perc_budget_30},"
                      f"{perc_budget_40},{perc_budget_50},{perc_budget_60},{perc_budget_70},{perc_budget_80},"
                      f"{perc_budget_90},{perc_budget_100},{avg_battery_perc}\n"
                      )

        filename = self.results_dir + self.__repr__() + "_" + self.sim_id + "_" + str(
            self.sim_idx) + f"_batteries_per_client_frac_{self.fit_fraction}.csv"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'a') as log:
            write = csv.writer(log)
            write.writerow(batteries_perc)

        return filtered_selected_cids

    def transpassed_budget(self):
        battery_perc = []
        perc_budget_10 = 0
        perc_budget_20 = 0
        perc_budget_30 = 0
        perc_budget_40 = 0
        perc_budget_50 = 0
        perc_budget_60 = 0
        perc_budget_70 = 0
        perc_budget_80 = 0
        perc_budget_90 = 0
        perc_budget_100 = 0
        for cid in self.clients_info:
            depletion = 1 - self.clients_info[cid]['battery'] / self.clients_info[cid]['initial_battery']
            battery_perc.append(self.clients_info[cid]['battery'] / self.clients_info[cid]['initial_battery'])

            if not self.clients_info[cid]['perc_budget_10'] and depletion > 0.1:
                self.clients_info[cid]['perc_budget_10'] = True
            if not self.clients_info[cid]['perc_budget_20'] and depletion > 0.2:
                self.clients_info[cid]['perc_budget_20'] = True
            if not self.clients_info[cid]['perc_budget_30'] and depletion > 0.3:
                self.clients_info[cid]['perc_budget_30'] = True
            if not self.clients_info[cid]['perc_budget_40'] and depletion > 0.4:
                self.clients_info[cid]['perc_budget_40'] = True
            if not self.clients_info[cid]['perc_budget_50'] and depletion > 0.5:
                self.clients_info[cid]['perc_budget_50'] = True
            if not self.clients_info[cid]['perc_budget_60'] and depletion > 0.6:
                self.clients_info[cid]['perc_budget_60'] = True
            if not self.clients_info[cid]['perc_budget_70'] and depletion > 0.7:
                self.clients_info[cid]['perc_budget_70'] = True
            if not self.clients_info[cid]['perc_budget_80'] and depletion > 0.8:
                self.clients_info[cid]['perc_budget_80'] = True
            if not self.clients_info[cid]['perc_budget_90'] and depletion > 0.9:
                self.clients_info[cid]['perc_budget_90'] = True
            if not self.clients_info[cid]['perc_budget_100'] and depletion == 1.0:
                self.clients_info[cid]['perc_budget_100'] = True

            if self.clients_info[cid]['perc_budget_10']:
                perc_budget_10 += 1
            if self.clients_info[cid]['perc_budget_20']:
                perc_budget_20 += 1
            if self.clients_info[cid]['perc_budget_30']:
                perc_budget_30 += 1
            if self.clients_info[cid]['perc_budget_40']:
                perc_budget_40 += 1
            if self.clients_info[cid]['perc_budget_50']:
                perc_budget_50 += 1
            if self.clients_info[cid]['perc_budget_60']:
                perc_budget_60 += 1
            if self.clients_info[cid]['perc_budget_70']:
                perc_budget_70 += 1
            if self.clients_info[cid]['perc_budget_80']:
                perc_budget_80 += 1
            if self.clients_info[cid]['perc_budget_90']:
                perc_budget_90 += 1
            if self.clients_info[cid]['perc_budget_100']:
                perc_budget_100 += 1
        return perc_budget_10, perc_budget_100, perc_budget_20, perc_budget_30, perc_budget_40, perc_budget_50, \
            perc_budget_60, perc_budget_70, perc_budget_80, perc_budget_90, sum(battery_perc) / len(battery_perc), \
            battery_perc

    def update_sample(self, client_manager, selected_cids):
        selected_clients = []

        for cid in selected_cids:
            selected_clients.append(client_manager.clients[str(cid)])

        return selected_clients

    # def decrease_battery(self, selected_cids):
    #     for cid in selected_cids:
    #         infer_time = self.clients_info[cid]['inference_latency']
    #         comm_time = self.clients_info[cid]['comm_latency']
    #         avg_joules = self.clients_info[cid]['avg_joules']
    #
    #         energy_consumed = get_energy_by_completion_time(infer_time, comm_time, avg_joules)
    #
    #         self.clients_info[cid]['battery'] -= energy_consumed