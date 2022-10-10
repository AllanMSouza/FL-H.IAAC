import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

NROWS = 3
NCOLS = 4

NCLIENTS = (10, 25, 50, 100)
MODELS   = ('Logist Regression', 'DNN', 'CNN')
ROUNDS   = 300

REMOVE = []

SOLUTIONS = os.listdir('.')
for solution in SOLUTIONS:
    if 'SGD' not in solution:
        REMOVE.append(solution)

for rem in REMOVE:
    SOLUTIONS.remove(rem)

def plot_evaluation(metric, estimator, xlabel, ylabel):

    fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(25, 17))

    for sol in SOLUTIONS:
        for model in MODELS:
            for clients in NCLIENTS:
                df = pd.read_csv(f'{sol}/{clients}/{model}/MNIST/evaluate_client.csv', 
                                 names=['round', 'id', 'parameters', 'loss', 'acc'])

                idx_clients = NCLIENTS.index(clients)
                idx_model   = MODELS.index(model)
                
                if estimator == 'mean':
                    df_client   = df.groupby(['round']).mean()
                    
                else:
                    df_client   = df.groupby(['round']).sum()
                    
                ax[idx_model, idx_clients].plot(range(len(df_client)),
                                                df_client[metric], 
                                                label=sol)

                ax[idx_model, idx_clients].grid(True, linestyle=':')
                ax[idx_model, idx_clients].legend()

                ax[idx_model, idx_clients].set_title(f'MODEL={model} NCLIENTS={clients}', fontweight='bold')
                ax[idx_model, idx_clients].set_xlabel(f'{xlabel}', size=14)
                ax[idx_model, idx_clients].set_ylabel(f'{ylabel}', size=14)



def plot_server(xlabel, ylabel):
    fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(25, 17))

    for sol in SOLUTIONS:
        for model in MODELS:
            for clients in NCLIENTS:
                df = pd.read_csv(f'{sol}/{clients}/{model}/MNIST/server.csv', 
                                 names=['round', 'avg_acc', 'top-3_acc', 'best_acc'])
                
                idx_clients = NCLIENTS.index(clients)
                idx_model   = MODELS.index(model)
                
                ax[idx_model, idx_clients].plot(df['round'], 
                                                 df['avg_acc'], 
                                                 label=sol)
                
                ax[idx_model, idx_clients].grid(True, linestyle=':')
                ax[idx_model, idx_clients].legend()
                ax[idx_model, idx_clients].set_xlim(0, ROUNDS)
                ax[idx_model, idx_clients].set_ylim(0.1, 1)
                
                ax[idx_model, idx_clients].set_title(f'MODEL={model} NCLIENTS={clients}', fontweight='bold')
                ax[idx_model, idx_clients].set_xlabel(f'{xlabel}', size=14)
                ax[idx_model, idx_clients].set_ylabel(f'{ylabel}', size=14)

    plt.show()
                
            


def plot_train(metric, estimator, xlabel, ylabel):

    fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(25, 17))

    for sol in SOLUTIONS:
        for model in MODELS:
            for clients in NCLIENTS:
                df = pd.read_csv(f'{sol}/{clients}/{model}/MNIST/train_client.csv', 
                                 names=['round', 'id', 'selected', 'total_time', 'parameters',
                                       'loss', 'acc'])

                idx_clients = NCLIENTS.index(clients)
                idx_model   = MODELS.index(model)
                
                if estimator == 'mean':
                    df_client   = df.groupby(['round']).mean()
                    
                else:
                    df_client   = df.groupby(['round']).sum()
                    
                ax[idx_model, idx_clients].plot(df_client.index, 
                                                df_client[metric], 
                                                label=sol)


                ax[idx_model, idx_clients].grid(True, linestyle=':')
                ax[idx_model, idx_clients].legend()

                ax[idx_model, idx_clients].set_title(f'MODEL={model} NCLIENTS={clients}', fontweight='bold')
                ax[idx_model, idx_clients].set_xlabel(f'{xlabel}', size=14)
                ax[idx_model, idx_clients].set_ylabel(f'{ylabel}', size=14)




def get_values_order_by(sol, clients, model, keys, estimator):
    df = pd.read_csv(f'{sol}/{clients}/{model}/MNIST/train_client.csv', 
                                 names=['round', 'id', 'selected', 'total_time', 'parameters',
                                       'loss', 'acc'])
    if estimator == 'mean':
        return df.groupby(keys).mean()

    if estimator == 'sum':
        return df.groupby(keys).sum()

def plot_metric_by_numberclients(metric, xlabel, ylabel):
    import warnings
    warnings.filterwarnings('ignore')

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))

    for sol in SOLUTIONS:
        for model in MODELS:
            df = pd.DataFrame()
            for clients in NCLIENTS:
                df_temp = pd.read_csv(f'{sol}/{clients}/{model}/MNIST/train_client.csv', 
                                 names=['round', 'id', 'selected', 'total_time', 'parameters',
                                       'loss', 'acc'])
                df_temp['clients'] = clients
                df = df.append(df_temp)
                
            idx_model = MODELS.index(model)
            df.reset_index(inplace=True)
            sns.lineplot('clients', metric, estimator='sum', data=df, label=sol, ax=ax[idx_model], 
                         markevery=1, marker='o', markersize=8, linewidth=2)
            ax[idx_model].grid(True, linestyle=':')
            ax[idx_model].set_title(f'Model: {model}', fontweight='bold')
            ax[idx_model].set_xlabel(f'{xlabel}', size=14)
            ax[idx_model].set_ylabel(f'{ylabel}', size=14)


