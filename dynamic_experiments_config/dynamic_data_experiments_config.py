import pandas as pd
import numpy as np


def change_pattern(n_patterns, n_clients, seed):

    client_pattern_dict = {i: i for i in range(n_clients)}

    np.random.seed(seed)
    # patterns = np.random.random_integers(low=0, high=n_patterns-1, size=n_clients)
    patterns = np.random.choice(range(n_patterns), n_clients, replace=False)
    # print(len(pd.Series(patterns).unique().tolist()))

    for i in range(len(client_pattern_dict)):

        client_id = list(client_pattern_dict.keys())[i]

        client_pattern_dict[client_id] = patterns[i]

    return client_pattern_dict

if __name__ == "__main__":

    n_rounds = 100
    n_clients = 20
    n_patterns = n_clients

    clients_ids = []
    rounds = []
    pattern = []

    rounds_to_change_pattern = [70]
    client_pattern_dict = {i: i for i in range(n_clients)}

    for i in range(1, n_rounds + 1):

        if i in rounds_to_change_pattern:

            client_pattern_dict = change_pattern(n_patterns, n_clients, i)

        for j in range(n_clients):

            rounds.append(i)
            clients_ids.append(j)
            pattern.append(client_pattern_dict[j])

    df = pd.DataFrame({'Round': rounds, 'Cid': clients_ids, 'Pattern': pattern})

    df.to_csv("dynamic_data_config.csv", index=False)


