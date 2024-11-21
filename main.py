import argparse
import json
import copy
import math
import random
import pandas as pd
import numpy as np
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import dataset_division
import FL_FedAvg
import FL_FedProx
import FL_FedUAB
import FL_pFedBayes

if __name__ == '__main__':

	# Read the hyperparameters stored in conf.json
	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	with open(args.conf, 'r') as f:
		conf = json.load(f)
	print('hyperparameters are: ',conf)

	# Use GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	# read dateset
	train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index = dataset_division.get_dataset('data/', conf)
	print('Dateset is OK.')

	# fedavg = FL_FedAvg.FedAvgServer(conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index)
	# fedavg.global_train()
	#
	# fedprox = FL_FedProx.FedProxServer(conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index)
	# fedprox.global_train()

	feduab1 = FL_FedUAB.FedUABServer(conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index, MonteCarlo_times=1)
	feduab1.global_train()

	feduab2 = FL_FedUAB.FedUABServer(conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index, MonteCarlo_times=2)
	feduab2.global_train()

	feduab5 = FL_FedUAB.FedUABServer(conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index, MonteCarlo_times=5)
	feduab5.global_train()

	pFedBayes = FL_pFedBayes.pFedBayesServer(conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index, MonteCarlo_times=5)
	pFedBayes.global_train()