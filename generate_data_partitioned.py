import flwr as fl
import tensorflow as tf
from optparse import OptionParser
import pickle
import subprocess
import logging
import os
import sys
import numpy as np

def cifar10(base_dir, train, test, partitined_dir, cid):
	with open(train, 'rb') as handle:
		idx_train = pickle.load(handle)

	with open(test, 'rb') as handle:
		idx_test = pickle.load(handle)

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	y_train = np.array([i[0] for i in y_train])
	y_test = np.array([i[0] for i in y_test])
	x_train, x_test = x_train / 255.0, x_test / 255.0

	x_train = x_train[idx_train]
	x_test = x_test[idx_test]
	y_train = y_train[idx_train]
	y_test = y_test[idx_test]
	with open("""{}{}_x_train.pickle""".format(partitined_dir, cid), 'wb') as f:
		pickle.dump(x_train, f)
	with open("""{}{}_x_test.pickle""".format(partitined_dir, cid), 'wb') as f:
		pickle.dump(x_test, f)
	with open("""{}{}_y_train.pickle""".format(partitined_dir, cid), 'wb') as f:
		pickle.dump(y_train, f)
	with open("""{}{}_y_test.pickle""".format(partitined_dir, cid), 'wb') as f:
		pickle.dump(y_test, f)

def main():
	parser = OptionParser()

	# parser.add_option("-a", "--algorithm", dest="algorithm", default='None',   help="Algorithm used for selecting clients", metavar="STR")
	parser.add_option("",   "--dataset",   dest="dataset",   default='MNIST',  help="")
	parser.add_option("", "--clients", dest="clients", default='10', help="")

	(opt, args) = parser.parse_args()

	clients = int(opt.clients)
	dataset = opt.dataset

	base_dir = """data/{}/{}/""".format(dataset, clients)

	partitioned_dir = """{}partitioned/""".format(base_dir)

	if dataset == 'CIFAR10':
		for i in range(clients):
			read_file_train = """{}idx_train_{}.pickle""".format(base_dir, i)
			read_file_test = """{}idx_test_{}.pickle""".format(base_dir, i)
			cifar10(base_dir, read_file_train, read_file_test, partitioned_dir, i)

		
		





if __name__ == '__main__':
	main()
