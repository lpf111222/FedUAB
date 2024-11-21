import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random
import pandas as pd
import numpy as np


class ClientBase(object):
    def __init__(self, conf, train_dataset, eval_dataset, data_distribution, train_index, eval_index, id):
        self.conf = conf
        self.client_id = id
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_distribution = data_distribution
        self.train_data_indices_lists = copy.deepcopy(train_index)
        self.train_data_quantity = len(train_index)
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.conf["batch_size"], sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       self.train_data_indices_lists))
        self.eval_data_indices_lists = copy.deepcopy(eval_index)
        self.eval_data_quantity = len(eval_index)
        self.eval_data_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.conf["batch_size"], sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       self.eval_data_indices_lists))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # This function is called only for the client to display the data label
        self.print_client_label()
        return None

    def personal_model_eval(self, model_parameters):
        # The client is tested on the same distribution of test data as the training
        with torch.no_grad():
            # The global model parameter Settings issued by the server are entered into the local model
            self.local_model.load_state_dict(model_parameters)
            # Set the model in eval mode
            self.local_model.eval()
            eval_loss = 0.0
            correct_num = 0
            for test_batch_id, test_batch in enumerate(self.eval_data_loader):
                test_X = test_batch[0].to(self.device)
                test_Y = test_batch[1].to(self.device)
                # Monte Carlo sampling
                logits_list = []
                for _ in range(self.MonteCarlo_times):
                    logits_list.append(self.local_model(test_X))
                logits_stack = torch.stack(logits_list)
                # p_hat is p^，the probability of each prediction
                p_hat_softmax = F.softmax(logits_stack, dim=2)
                # # p_bar is p-，the average of multiple probabilities
                p_bar_softmax = p_hat_softmax.mean(dim=0)
                # Predicting labels
                pred_Y = p_bar_softmax.max(1)[1]
                # loss
                eval_loss += F.nll_loss(torch.log(p_bar_softmax), test_Y, reduction='sum').item()
                # Count the correct number
                correct_num += pred_Y.eq(test_Y.view_as(pred_Y)).sum().item()
            # acc , loss
            eval_acc = correct_num / self.eval_data_quantity * 100.0
            eval_loss /= self.eval_data_quantity
            # print
            print("client %d，finished personalized model test，personalized eval_acc：%f，personalized eval_loss：%f" % (self.client_id, eval_acc, eval_loss))
        return eval_acc, eval_loss, self.eval_data_quantity

    def print_client_label(self):
        # This function is called only for the client to display the data label
        y_label_list = []
        if self.conf['dataset_name'] == 'mnist':
            all_label = self.train_dataset.targets.tolist()
        elif self.conf['dataset_name'] == 'cnews':
            all_label = self.train_dataset.targets
        # Count all labels
        for i in self.train_data_indices_lists:
            y_label_list.append(all_label[i])
        # Removing duplicate elements
        y_label_list = list(set(y_label_list))
        print('Client %d has number of training examples: %d, label class: %s' % (self.client_id, self.train_data_quantity,', '.join('%s' % id for id in y_label_list)))
        return None