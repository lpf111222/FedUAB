import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random
import pandas as pd
import numpy as np


class ServerBase(object):

    def __init__(self, conf, eval_dataset):
        self.conf = conf
        # eval dataset
        self.eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
        self.eval_data_quantity = len(eval_dataset.targets)
        self.global_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The training and testing information was recorded and saved as excel after completing the training
        self.train_eval_records = pd.DataFrame(index=list(range(self.conf['global_epochs'])),
                                               columns=['train_NLL_loss', 'train_KL_loss',
                                                        'global_eval_acc','global_eval_loss',
                                                        'fine_tuning_personal_eval_acc',
                                                        'fine_tuning_personal_eval_loss'],
                                               dtype=float)
        return None

    def global_train(self):
        for e in range(self.conf['global_epochs']):
            # broadcast, train, upload
            clients_trained_parameters, train_data_quantity_sum = self.broadcast_train_upload()
            # Server model weight aggregation
            new_model_parameters = self.aggregate_parameters(clients_trained_parameters, train_data_quantity_sum)
            # Save the aggregated server model parameters
            self.global_model.load_state_dict(new_model_parameters)
            # Testing of the server global model and the client personalized model.
            # It is computationally expensive and takes place only once in every N epochs. The figure in the paper is N=10
            if e%100==0:
                self.global_model_eval()
                self.all_clients_personal_model_eval()
                self.save_records()
            # Finish
            self.global_epoch += 1
        return None

    def broadcast_train_upload(self):
        # The current model parameters, a copy, next sent to the client
        old_model_parameters = copy.deepcopy(self.global_model.state_dict())
        # k clients are randomly selected to participate in the training
        # Set the random seed so that the same client is selected for each experiment
        random.seed(self.global_epoch)
        training_clients_list = random.sample(self.clients, self.conf['k'])
        clients_trained_parameters = []
        train_data_quantity_sum = 0
        train_NLL_loss_sum = 0.0
        train_KL_loss_sum = 0.0
        # Training one by one
        for c in training_clients_list:
            trained_model_parameters, train_data_quantity, train_NLL_loss, train_KL_loss = c.local_training(old_model_parameters=old_model_parameters)
            # The model parameters uploaded by the client are stored in a list
            clients_trained_parameters.append({'trained_model_parameters': trained_model_parameters, 'train_data_quantity': train_data_quantity})
            train_data_quantity_sum += train_data_quantity
            train_NLL_loss_sum += (train_NLL_loss * train_data_quantity)
            train_KL_loss_sum += (train_KL_loss * train_data_quantity)
        # Recording the training loss
        train_NLL_loss = train_NLL_loss_sum / train_data_quantity_sum
        train_KL_loss = train_KL_loss_sum / train_data_quantity_sum
        self.train_eval_records['train_NLL_loss'][self.global_epoch] = train_NLL_loss
        self.train_eval_records['train_KL_loss'][self.global_epoch] = train_KL_loss
        # print
        print('The server completes round %d of global training，NLL loss：%f，KL loss：%f，' % (self.global_epoch, train_NLL_loss, train_KL_loss))
        return clients_trained_parameters, train_data_quantity_sum

    def aggregate_parameters(self, clients_trained_parameters, train_data_quantity_sum):
        # create an empty dictionary ready for our new arguments
        new_model_parameters = dict()
        for name, params in clients_trained_parameters[0]['trained_model_parameters'].items():
            new_model_parameters[name] = torch.zeros_like(params)
        # The weighted summation proposed by FedAvg
        for c in clients_trained_parameters:
            # The weight coefficient is calculated according to the amount of client data
            client_weight = c['train_data_quantity'] / train_data_quantity_sum
            for name, params in c['trained_model_parameters'].items():
                new_model_parameters[name].add_(params, alpha=client_weight)
        return new_model_parameters

    def global_model_eval(self):    # Server global model test
        with torch.no_grad():
            self.global_model.eval()
            global_correct_num = 0
            global_eval_loss_num = 0.0
            for test_batch_id, test_batch in enumerate(self.eval_data_loader):
                test_X = test_batch[0].to(self.device)
                test_Y = test_batch[1].to(self.device)
                # Monte Carlo sampling
                logits_list = []
                for _ in range(self.MonteCarlo_times):
                    logits_list.append(self.global_model(test_X))
                logits_stack = torch.stack(logits_list)
                # p_hat is p^，the probability of each prediction
                p_hat_softmax = F.softmax(logits_stack, dim=2)
                # p_bar is p-，the average of multiple probabilities
                p_bar_softmax = p_hat_softmax.mean(dim=0)
                # Predicting labels
                pred_Y = p_bar_softmax.max(1)[1]
                # loss
                global_eval_loss_num += F.nll_loss(torch.log(p_bar_softmax), test_Y, reduction='sum').item()
                # Count the correct number
                global_correct_num += pred_Y.eq(test_Y.view_as(pred_Y)).sum().item()
            # acc , loss
            global_eval_acc = global_correct_num / self.eval_data_quantity * 100.0
            global_eval_loss = global_eval_loss_num / self.eval_data_quantity
            self.train_eval_records['global_eval_acc'][self.global_epoch] = global_eval_acc
            self.train_eval_records['global_eval_loss'][self.global_epoch] = global_eval_loss
            # print
            print("Server %s completes round %d of global model test，eval_acc：%f，eval_loss：%f"
                  % (self.server_name, self.global_epoch, global_eval_acc, global_eval_loss))
        return None

    def all_clients_personal_model_eval(self):  # Client Personalization Model test
        # The current model parameters, a copy, next sent to the client
        old_model_parameters = copy.deepcopy(self.global_model.state_dict())
        fine_tuning_personal_eval_correct_sum = 0
        fine_tuning_personal_eval_loss_sum = 0.0
        fine_tuning_personal_eval_data_quantity_sum = 0
        for c in self.clients:
            # Each client is fine-tuned locally before personalized testing
            trained_model_parameters, _, __, ___ = c.local_training(old_model_parameters)
            fine_tuning_client_eval_acc, fine_tuning_client_eval_loss, fine_tuning_client_eval_data_quantity = c.personal_model_eval(
                trained_model_parameters)
            fine_tuning_personal_eval_correct_sum += fine_tuning_client_eval_data_quantity * fine_tuning_client_eval_acc / 100.0
            fine_tuning_personal_eval_loss_sum += fine_tuning_client_eval_data_quantity * fine_tuning_client_eval_loss
            fine_tuning_personal_eval_data_quantity_sum += fine_tuning_client_eval_data_quantity
        # Statistically averaged acc and loss
        fine_tuning_personal_eval_acc = fine_tuning_personal_eval_correct_sum / fine_tuning_personal_eval_data_quantity_sum * 100.0
        fine_tuning_personal_eval_loss = fine_tuning_personal_eval_loss_sum / fine_tuning_personal_eval_data_quantity_sum
        # save
        self.train_eval_records['fine_tuning_personal_eval_acc'][self.global_epoch] = fine_tuning_personal_eval_acc
        self.train_eval_records['fine_tuning_personal_eval_loss'][self.global_epoch] = fine_tuning_personal_eval_loss
        # print
        print("server%s，complete the %d round of personalized test， personalized eval_acc：%f，personalized eval_loss：%f" % (
            self.server_name, self.global_epoch, fine_tuning_personal_eval_acc, fine_tuning_personal_eval_loss))
        return None

    def save_records(self):
        # Save model parameters
        model_parameters = self.global_model.state_dict()
        torch.save(model_parameters, 'result/' + self.server_name + '_params.pt')
        # The training and testing information was saved as excel
        excel_writer = pd.ExcelWriter('result/' + self.server_name + '_records.xlsx')
        self.train_eval_records.to_excel(excel_writer, index_label=False)
        excel_writer.close()
        return None