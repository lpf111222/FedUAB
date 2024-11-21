import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random
import pandas as pd
import numpy as np
import model_frequentist, model_BayebyBackprop, server_base, client_base


class FedAvgClient(client_base.ClientBase):
    def __init__(self, global_model, conf, train_dataset, eval_dataset, data_distribution, train_index, eval_index, id, MonteCarlo_times=1):
        super().__init__(conf, train_dataset, eval_dataset, data_distribution, train_index, eval_index, id)
        # Create client local model
        self.local_model = copy.deepcopy(global_model)
        # FedAvg MonteCarlo_times=1
        self.MonteCarlo_times = MonteCarlo_times

    def local_training(self, old_model_parameters):
        # The global model parameter Settings issued by the server are entered into the local model
        self.local_model.load_state_dict(old_model_parameters)
        # Set the model in training mode
        self.local_model.train()
        lr_client = self.conf["lr"]
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr_client)
        train_NLL_loss = 0.0
        # Local training epochs
        for e in range(self.conf["local_epochs"]):
            # The data for each epoch training is shuffled
            random.shuffle(self.train_data_indices_lists)
            for batch_id, batch in enumerate(self.train_data_loader):
                data = batch[0].to(self.device)
                target = batch[1].to(self.device)
                optimizer.zero_grad()
                logits = self.local_model(data)
                loss = F.cross_entropy(logits, target, reduction='mean')
                loss.backward()
                optimizer.step()
                train_NLL_loss += loss.item() * len(target)
        train_NLL_loss /= (self.conf["local_epochs"] * self.train_data_quantity)
        train_KL_loss = 0.0
        # Read the locally trained parameters
        trained_model_parameters = self.local_model.state_dict()
        # print
        print('Client %d completes local training, error loss: %f' % (self.client_id, train_NLL_loss))
        # Return: trained parameters, number of client data samples, loss
        return trained_model_parameters, self.train_data_quantity, train_NLL_loss, train_KL_loss


class FedAvgServer(server_base.ServerBase):
    def __init__(self, conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index, MonteCarlo_times=1):
        super().__init__(conf, eval_dataset)
        self.server_name = 'FedAvg'
        # Create a server model
        self.global_model = model_frequentist.get_model(self.conf["model_name"])
        # FedAvg MonteCarlo_times=1
        self.MonteCarlo_times = MonteCarlo_times
        # The client is  a member of the server
        self.clients = []
        for i in range(conf["num_client"]):
            self.clients.append(
                FedAvgClient(self.global_model, conf, train_dataset, eval_dataset, clients_distribution[i],
                             train_clients_index[i],
                             eval_clients_index[i], i))
        return None