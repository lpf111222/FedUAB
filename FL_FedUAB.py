import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random
import pandas as pd
import numpy as np
import model_frequentist, model_BayebyBackprop, server_base, client_base


class FedUABClient(client_base.ClientBase):
    def __init__(self, global_model, conf, train_dataset, eval_dataset, data_distribution, train_index, eval_index, id, MonteCarlo_times=1):
        super().__init__(conf, train_dataset, eval_dataset, data_distribution, train_index, eval_index, id)
        #  local model
        self.local_model = copy.deepcopy(global_model)
        self.MonteCarlo_times = MonteCarlo_times

    def local_training(self, old_model_parameters):
        # The global model parameters broadcast by the server are the local model parameters
        self.local_model.load_state_dict(old_model_parameters)
        # The current global model parameters should also be used as prior distributions
        prior_dict = dict()
        for name, params in old_model_parameters.items():
            if "rho" in name:
                # change rho to sigma using the softlpus function
                prior_dict[name.replace('rho', 'sigma')] = torch.log1p(torch.exp(params)).detach()
            else:
                # mu
                prior_dict[name] = params.clone().detach()
        self.local_model.set_prior(prior_dict=prior_dict)
        # Set the model in training mode
        self.local_model.train()
        lr_client = self.conf["lr"]
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr_client)
        train_NLL_loss = 0.0
        train_KL_loss = 0.0
        # Local training epochs
        for e in range(self.conf["local_epochs"]):
            # The data for each epoch training is shuffled
            random.shuffle(self.train_data_indices_lists)
            for batch_id, batch in enumerate(self.train_data_loader):
                data = batch[0].to(self.device)
                target = batch[1].to(self.device)
                optimizer.zero_grad()
                # Monte Carlo sampling
                for u in range(self.MonteCarlo_times):
                    logits = self.local_model(data)
                    one_error_loss = F.cross_entropy(logits, target, reduction='mean').clamp(max=100.0)
                    if u == 0:
                        error_loss_sum = one_error_loss
                    else:
                        error_loss_sum += one_error_loss
                # NLL loss
                error_loss = error_loss_sum / self.MonteCarlo_times
                # complexity_loss
                complexity_loss = self.local_model.get_kl() * self.conf["kl_division"]
                # loss
                loss = error_loss + complexity_loss
                loss.backward()
                optimizer.step()
                train_NLL_loss += error_loss.item() * len(target)
                train_KL_loss += complexity_loss.item() * len(target)
        train_NLL_loss /= (self.conf["local_epochs"] * self.train_data_quantity)
        train_KL_loss /= (self.conf["local_epochs"] * self.train_data_quantity)
        # Read the locally trained parameters
        trained_model_parameters = self.local_model.state_dict()
        print('Client %d completes local training, NLL loss: %f, KL loss %f' % (self.client_id, train_NLL_loss, train_KL_loss))
        # Return: trained parameters, number of client data samples, loss
        return trained_model_parameters, self.train_data_quantity, train_NLL_loss, train_KL_loss


class FedUABServer(server_base.ServerBase):
    def __init__(self, conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index, MonteCarlo_times=1):
        super().__init__(conf, eval_dataset)
        self.server_name = 'FedUAB-' + str(MonteCarlo_times)
        # Create a server model
        self.global_model = model_BayebyBackprop.get_model(self.conf["model_name"], variance_reduction=2)
        self.MonteCarlo_times = MonteCarlo_times
        # The client is  a member of the server
        self.clients = []
        for i in range(conf["num_client"]):
            self.clients.append(
                FedUABClient(self.global_model, conf, train_dataset, eval_dataset, clients_distribution[i],
                             train_clients_index[i],
                             eval_clients_index[i], i, MonteCarlo_times))
        return None

    def aggregate_parameters(self, clients_trained_parameters, train_data_quantity_sum):
        # The aggregation method for probability conflation in the Weights aggregation section of the paper
        new_model_parameters = dict()
        for name, params in clients_trained_parameters[0]['trained_model_parameters'].items():
            new_model_parameters[name] = torch.zeros_like(params)
        # Equation 8 of the paper
        for c in clients_trained_parameters:
            client_weight = c['train_data_quantity'] / train_data_quantity_sum
            for name, params in c['trained_model_parameters'].items():
                if "rho" in name:
                    name_for_mu = name.replace('rho', 'mu')
                    mu_temp = c['trained_model_parameters'][name_for_mu]
                    sigma_temp = torch.log1p(torch.exp(params))
                    new_model_parameters[name_for_mu].add_(torch.mul(mu_temp, sigma_temp ** (-2)), alpha=client_weight)
                    new_model_parameters[name].add_(sigma_temp ** (-2), alpha=client_weight)
        # After all clients are aggregated, the mean mu is calculated
        for name, params in new_model_parameters.items():
            if "mu" in name:
                name_for_rho = name.replace('mu', 'rho')
                params.div_(new_model_parameters[name_for_rho])
        # the standard deviation sigma is converted to the parameter rho
        for name, params in new_model_parameters.items():
            if "rho" in name:
                sigma = params.pow(-1) ** 0.5
                new_model_parameters[name] = torch.log(torch.exp(sigma) - 1.0)
        return new_model_parameters