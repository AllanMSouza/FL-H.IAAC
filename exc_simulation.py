import flwr as fl
import tensorflow as tf
from client import FedClient
from servidor import FedServer

NUM_CLIENTS = 2

def create_client(cid):
	return FedClient(cid, NUM_CLIENTS, model_name='DNN', client_selection=False)

strategy = FedServer(aggregation_method='None', fraction_fit=1, num_clients=NUM_CLIENTS)


fl.simulation.start_simulation(
    client_fn   = create_client,
    num_clients = NUM_CLIENTS,
    config      = fl.server.ServerConfig(num_rounds=2),
    strategy    = strategy,
)
