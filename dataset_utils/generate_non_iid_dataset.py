import argparse
import os
import random
import time

from utils import save_dataloaders

trainloaders = []
valloaders = []
cli_performances = []
testloader = None


def parse_arguments():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--sim_id", help="Simulation identifier", default=0)
    argParser.add_argument("--num_clients", help="Number of clients", default=10)
    argParser.add_argument("--batch_size", help="Batch size", default=32)
    argParser.add_argument("--data_dir", help="Dataset directory", default="dataset/")
    argParser.add_argument("--dataset", help="Options: Cifar10, Mnist", default="CIFAR10")
    argParser.add_argument("--num_classes", help="Number of dataset classes to use", default=10)
    argParser.add_argument("--balance", help="User dataset balanced", default=False)
    argParser.add_argument("--partition", help="Dataset partition: dir or pat", default="dir")
    argParser.add_argument("--class_per_client", help="Number of classes per client", default=10)
    argParser.add_argument("--train_perc", help="Dataset percentual for train", default=0.75)
    argParser.add_argument("--alpha_dir", help="Dirichlet alpha parameter", default=0.1)
    argParser.add_argument("--niid", help="If niid", default=True)

    args = argParser.parse_args()

    # os.environ["dataset"] = args.dataset
    # os.environ["num_classes"] = args.num_classes
    # os.environ["data_dir"] = args.data_dir

    return args


if __name__ == '__main__':
    # Random initializations
    random.seed(time.time())

    args = parse_arguments()

    save_dataloaders(args.dataset, int(args.num_clients),
                     int(args.num_classes),
                     bool(args.niid),
                     bool(args.balance), args.partition,
                     int(args.class_per_client),
                     int(args.batch_size),
                     float(args.train_perc),
                     float(args.alpha_dir),
                     args.data_dir,
                     int(args.sim_id)
                     )
