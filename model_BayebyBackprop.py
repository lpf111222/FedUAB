import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random
import pandas as pd
import numpy as np
import pickle


def get_model(name, variance_reduction=2):
    # device is GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a model based on model name
    if name == "fcn_mnist":
        model = BBB_FCN_Mnist(variance_reduction=variance_reduction).to(device)
    elif name == "cnn_cifar":
        model = BBB_CNN_Cifar(variance_reduction=variance_reduction).to(device)
    elif name == "text_cnn":
        model = BBB_TextCNN(variance_reduction=variance_reduction).to(device)
    else:
        print('Model_name is wrong!')
    return model

# Convolutional layers of Bayes by Backprop
class BBBConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, init_mu=0.1, init_sigma2=0.01):
        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kl = 0.0
        self.out_mu = 0.0
        self.out_var = 0.0
        # Parameter tensors, mu and rho
        self.W_mu = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_rho = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias_mu = torch.nn.Parameter(torch.Tensor(out_channels))
        self.bias_rho = torch.nn.Parameter(torch.Tensor(out_channels))
        # Initialize the parameters, mu is uniformly distributed, and rho is specified by the initial value
        self.W_mu.data.uniform_(-init_mu, init_mu)
        self.W_rho.data.fill_(math.log(math.exp(init_sigma2 ** 0.5) - 1.0))
        self.bias_mu.data.fill_(0.01)
        self.bias_rho.data.fill_(math.log(math.exp(init_sigma2 ** 0.5) - 1.0))
        # The prior probability is going to take the KL divergence from the variational posterior
        self.prior_W_mu = torch.zeros_like(self.W_mu, device=self.device)
        self.prior_W_sigma = torch.zeros_like(self.W_rho, device=self.device).fill_(init_sigma2 ** 0.5)
        self.prior_bias_mu = torch.zeros_like(self.bias_mu, device=self.device)
        self.prior_bias_sigma = torch.zeros_like(self.bias_rho, device=self.device).fill_(init_sigma2 ** 0.5)
        return None

    def forward(self, x, Mode='LocalReparametrisationTrick'):
        if Mode == 'SamplingWeight': # it can be sampled directly
            # sampling W, Limit to -10 to 10, extreme values may cause overflow
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            W_eps = torch.normal(mean=0, std=1, size=self.W_sigma.size()).to(self.device).clamp(min=-10, max=10)
            sampled_W = self.W_mu + self.W_sigma * W_eps
            # sampling bias, Limit to -10 to 10, extreme values may cause overflow
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_eps = torch.normal(mean=0, std=1, size=self.bias_sigma.size()).to(self.device).clamp(min=-10, max=10)
            sampled_bias = self.bias_mu + self.bias_sigma * bias_eps
            # forword
            out = F.conv2d(x, sampled_W, sampled_bias, self.stride, self.padding, self.dilation, self.groups)
        elif Mode == 'LocalReparametrisationTrick': # we use Local Reparametrisation Trick
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            # mean mu
            self.out_mu = F.conv2d(x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)
            # variance
            self.out_var = 1e-16 + F.conv2d(x ** 2, self.W_sigma ** 2, self.bias_sigma ** 2, self.stride, self.padding, self.dilation, self.groups)
            # standard deviation
            self.out_sigma = self.out_var.sqrt()
            # Sampling eps from a normal distribution N(0,1)
            eps = torch.normal(mean=0, std=1, size=self.out_mu.size()).to(self.device).clamp(min=-10, max=10)
            # out = mu + standard deviation * eps
            out = self.out_mu + self.out_sigma * eps
        else:
            print('forward mode is wrong.')
        return out

    def get_kl(self):
        # kl_divergence
        self.kl = torch.distributions.kl_divergence(torch.distributions.Normal(self.W_mu, self.W_sigma),
                                                    torch.distributions.Normal(self.prior_W_mu,
                                                                               self.prior_W_sigma)).sum()
        self.kl += torch.distributions.kl_divergence(torch.distributions.Normal(self.bias_mu, self.bias_sigma),
                                                     torch.distributions.Normal(self.prior_bias_mu,
                                                                                self.prior_bias_sigma)).sum()
        return self.kl


# Linear layers of Bayes by Backprop
class BBBLinear(nn.Module):
    def __init__(self, in_features, out_features, init_mu=0.1, init_sigma2=0.01):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kl = None
        self.out_mu = None
        self.out_sigma = None
        self.out_var = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Parameter tensors, mu and rho
        self.W_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = torch.nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = torch.nn.Parameter(torch.Tensor(out_features))
        # Initialize the parameters, mu is uniformly distributed, and rho is specified by the initial value
        self.W_mu.data.uniform_(-init_mu, init_mu)
        self.W_rho.data.fill_(math.log(math.exp(init_sigma2 ** 0.5) - 1.0))
        self.bias_mu.data.fill_(0.01)
        self.bias_rho.data.fill_(math.log(math.exp(init_sigma2 ** 0.5) - 1.0))
        # The prior probability is going to take the KL divergence from the variational posterior
        self.prior_W_mu = torch.zeros_like(self.W_mu, device=self.device)
        self.prior_W_sigma = torch.zeros_like(self.W_rho, device=self.device).fill_(init_sigma2 ** 0.5)
        self.prior_bias_mu = torch.zeros_like(self.bias_mu, device=self.device)
        self.prior_bias_sigma = torch.zeros_like(self.bias_rho, device=self.device).fill_(init_sigma2 ** 0.5)
        return None

    def forward(self, x, Mode='LocalReparametrisationTrick'):
        if Mode == 'SamplingWeight':    # it can be sampled directly
            # sampling W, Limit to -10 to 10, extreme values may cause overflow
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            W_eps = torch.normal(mean=0, std=1, size=self.W_sigma.size()).to(self.device).clamp(min=-10, max=10)
            sampled_W = self.W_mu + self.W_sigma * W_eps
            # sampling bias, Limit to -10 to 10, extreme values may cause overflow
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_eps = torch.normal(mean=0, std=1, size=self.bias_sigma.size()).to(self.device).clamp(min=-10, max=10)
            sampled_bias = self.bias_mu + self.bias_sigma * bias_eps
            # forword
            out = F.linear(x, sampled_W, sampled_bias)
        elif Mode == 'LocalReparametrisationTrick':     # we use Local Reparametrisation Trick
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            # mean mu
            self.out_mu = F.linear(x, self.W_mu, self.bias_mu)
            # variance
            self.out_var = 1e-16 + F.linear(x ** 2, self.W_sigma ** 2, self.bias_sigma ** 2)
            # standard deviation
            self.out_sigma = self.out_var.sqrt()
            # Sampling eps from a normal distribution N(0,1)
            eps = torch.normal(mean=0, std=1, size=self.out_mu.size()).to(self.device).clamp(min=-10, max=10)
            # out = mu + standard deviation * eps
            out = self.out_mu + self.out_sigma * eps
        else:
            print('forward mode is wrong.')
        return out

    def get_kl(self):
        # kl_divergence
        self.kl = torch.distributions.kl_divergence(torch.distributions.Normal(self.W_mu, self.W_sigma),
                                                    torch.distributions.Normal(self.prior_W_mu,
                                                                               self.prior_W_sigma)).sum()
        self.kl += torch.distributions.kl_divergence(torch.distributions.Normal(self.bias_mu, self.bias_sigma),
                                                     torch.distributions.Normal(self.prior_bias_mu,
                                                                                self.prior_bias_sigma)).sum()
        return self.kl


# Embedding layers of Bayes by Backprop
class BBBEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init_mu=0.1, init_sigma2=0.01):
        super(BBBEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.kl = 0.0
        self.out_mu = 0.0
        self.out_var = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Parameter tensors, mu and rho
        self.W_mu = torch.nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.W_rho = torch.nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        # Initialize the parameters, mu is uniformly distributed, and rho is specified by the initial value
        self.W_mu.data.uniform_(-init_mu, init_mu)
        self.W_rho.data.fill_(math.log(math.exp(init_sigma2 ** 0.5) - 1.0))
        # The prior probability is going to take the KL divergence from the variational posterior
        self.prior_W_mu = torch.zeros_like(self.W_mu, device=self.device)
        self.prior_W_sigma = torch.zeros_like(self.W_rho, device=self.device).fill_(init_sigma2 ** 0.5)
        return None

    def forward(self, x, Mode='SamplingWeight'):
        # Embeddings can only be sampled directly
        if Mode == 'SamplingWeight':
            # sampling W, Limit to -10 to 10, extreme values may cause overflow
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            eps = torch.normal(mean=0, std=1, size=self.W_sigma.size()).to(self.device).clamp(min=-10, max=10)
            sampled_weights = self.W_mu + self.W_sigma * eps
            # forword
            out = F.embedding(x, sampled_weights)
        else:
            print('forward mode is wrong.')
        return out

    def get_kl(self):
        # kl_divergence
        self.kl = torch.distributions.kl_divergence(torch.distributions.Normal(self.W_mu, self.W_sigma),
                                                    torch.distributions.Normal(self.prior_W_mu,
                                                                               self.prior_W_sigma)).sum()
        return self.kl


# fcn_mnist model for Mnist
class BBB_FCN_Mnist(nn.Module):
    def __init__(self, variance_reduction=2):
        super(BBB_FCN_Mnist, self).__init__()
        # The variance init_sigma2 decreases layer by layer
        self.fc1 = BBBLinear(784, 50, init_mu=0.25, init_sigma2=0.01)
        self.fc2 = BBBLinear(50, 50, init_mu=0.15, init_sigma2=0.01/variance_reduction)
        self.fc3 = BBBLinear(50, 10, init_mu=0.1, init_sigma2=0.01/(variance_reduction**2))
        return None

    def forward(self, input):
        x = input.view(input.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_kl(self):
        return self.fc1.get_kl() + self.fc2.get_kl() + self.fc3.get_kl()

    def set_prior(self, prior_dict=None):
        # Modifying the prior，fc1
        self.fc1.prior_W_mu = prior_dict['fc1.W_mu']
        self.fc1.prior_W_sigma = prior_dict['fc1.W_sigma']
        self.fc1.prior_bias_mu = prior_dict['fc1.bias_mu']
        self.fc1.prior_bias_sigma = prior_dict['fc1.bias_sigma']
        # Modifying the prior，fc2
        self.fc2.prior_W_mu = prior_dict['fc2.W_mu']
        self.fc2.prior_W_sigma = prior_dict['fc2.W_sigma']
        self.fc2.prior_bias_mu = prior_dict['fc2.bias_mu']
        self.fc2.prior_bias_sigma = prior_dict['fc2.bias_sigma']
        # Modifying the prior，fc3
        self.fc3.prior_W_mu = prior_dict['fc3.W_mu']
        self.fc3.prior_W_sigma = prior_dict['fc3.W_sigma']
        self.fc3.prior_bias_mu = prior_dict['fc3.bias_mu']
        self.fc3.prior_bias_sigma = prior_dict['fc3.bias_sigma']
        return None

# cnn_cifar model for Cifar10
class BBB_CNN_Cifar(nn.Module):
    def __init__(self, variance_reduction=2):
        super(BBB_CNN_Cifar, self).__init__()
        # The variance init_sigma2 decreases layer by layer
        self.conv1 = BBBConv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=0, stride=1, init_mu=0.20, init_sigma2=0.01)
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.conv2 = BBBConv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, stride=1, init_mu=0.18, init_sigma2=0.01/variance_reduction)
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.conv3 = BBBConv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, stride=1, init_mu=0.15, init_sigma2=0.01/(variance_reduction**2))
        self.fc1 = BBBLinear(32 * 4 * 4, 64, init_mu=0.12, init_sigma2=0.01/(variance_reduction**3))
        self.fc2 = BBBLinear(64, 10, init_mu=0.10, init_sigma2=0.01/(variance_reduction**4))
        return None

    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_kl(self):
        return self.conv1.get_kl() + self.conv2.get_kl() + self.conv3.get_kl() + self.fc1.get_kl() + self.fc2.get_kl()

    def set_prior(self, prior_dict=None):
        # Modifying the prior，conv1
        self.conv1.prior_W_mu = prior_dict['conv1.W_mu']
        self.conv1.prior_W_sigma = prior_dict['conv1.W_sigma']
        self.conv1.prior_bias_mu = prior_dict['conv1.bias_mu']
        self.conv1.prior_bias_sigma = prior_dict['conv1.bias_sigma']
        # Modifying the prior，conv2
        self.conv2.prior_W_mu = prior_dict['conv2.W_mu']
        self.conv2.prior_W_sigma = prior_dict['conv2.W_sigma']
        self.conv2.prior_bias_mu = prior_dict['conv2.bias_mu']
        self.conv2.prior_bias_sigma = prior_dict['conv2.bias_sigma']
        # Modifying the prior，conv3
        self.conv3.prior_W_mu = prior_dict['conv3.W_mu']
        self.conv3.prior_W_sigma = prior_dict['conv3.W_sigma']
        self.conv3.prior_bias_mu = prior_dict['conv3.bias_mu']
        self.conv3.prior_bias_sigma = prior_dict['conv3.bias_sigma']
        # Modifying the prior，fc1
        self.fc1.prior_W_mu = prior_dict['fc1.W_mu']
        self.fc1.prior_W_sigma = prior_dict['fc1.W_sigma']
        self.fc1.prior_bias_mu = prior_dict['fc1.bias_mu']
        self.fc1.prior_bias_sigma = prior_dict['fc1.bias_sigma']
        # Modifying the prior，fc2
        self.fc2.prior_W_mu = prior_dict['fc2.W_mu']
        self.fc2.prior_W_sigma = prior_dict['fc2.W_sigma']
        self.fc2.prior_bias_mu = prior_dict['fc2.bias_mu']
        self.fc2.prior_bias_sigma = prior_dict['fc2.bias_sigma']
        return None

# text_cnn model for cnews
class BBB_TextCNN(nn.Module):
    def __init__(self, vocab_size=50000, embedding_dim=16, num_filter=10,
                 filter_sizes=[2,3,4], output_dim=10, variance_reduction=2):
        super(BBB_TextCNN, self).__init__()
        # The variance init_sigma2 decreases layer by layer
        # embedding，embed a vocabulary of 50000 words into a 32-dimensional
        self.embedding = BBBEmbedding(vocab_size, embedding_dim, init_mu=0.50, init_sigma2=0.01)
        # Convolutional layers, all num_filter (100) channels, volumes of length 2,3,4
        self.conv1 = BBBConv2d(in_channels=1, out_channels=num_filter, kernel_size=(filter_sizes[0], embedding_dim), init_mu=0.30, init_sigma2=0.01/variance_reduction)
        self.conv2 = BBBConv2d(in_channels=1, out_channels=num_filter, kernel_size=(filter_sizes[1], embedding_dim), init_mu=0.30, init_sigma2=0.01/variance_reduction)
        self.conv3 = BBBConv2d(in_channels=1, out_channels=num_filter, kernel_size=(filter_sizes[2], embedding_dim), init_mu=0.30, init_sigma2=0.01/variance_reduction)
        # 300 -> 10
        self.fc = BBBLinear(len(filter_sizes) * num_filter, output_dim, init_mu=0.10, init_sigma2=0.01/(variance_reduction**2))
        return None

    def forward(self, input):
        embedded = self.embedding(input)
        unsqueeze_embedded = embedded.unsqueeze(1)
        conv1_ed = F.leaky_relu(self.conv1(unsqueeze_embedded).squeeze(3))
        conv2_ed = F.leaky_relu(self.conv2(unsqueeze_embedded).squeeze(3))
        conv3_ed = F.leaky_relu(self.conv3(unsqueeze_embedded).squeeze(3))
        maxpooled1 = conv1_ed.max(dim=2)[0]
        maxpooled2 = conv2_ed.max(dim=2)[0]
        maxpooled3 = conv3_ed.max(dim=2)[0]
        # concatenating the three Max pools, into the fully connected layer
        return self.fc(torch.cat([maxpooled1, maxpooled2, maxpooled3], dim=1))

    def get_kl(self):
        return self.embedding.get_kl() + self.conv1.get_kl() + self.conv2.get_kl() + self.conv3.get_kl() + self.fc.get_kl()

    def set_prior(self, prior_dict=None):
        # Modifying the prior，embedding
        self.embedding.prior_W_mu = prior_dict['embedding.W_mu']
        self.embedding.prior_W_sigma = prior_dict['embedding.W_sigma']
        # Modifying the prior，conv1
        self.conv1.prior_W_mu = prior_dict['conv1.W_mu']
        self.conv1.prior_W_sigma = prior_dict['conv1.W_sigma']
        self.conv1.prior_bias_mu = prior_dict['conv1.bias_mu']
        self.conv1.prior_bias_sigma = prior_dict['conv1.bias_sigma']
        # Modifying the prior，conv2
        self.conv2.prior_W_mu = prior_dict['conv2.W_mu']
        self.conv2.prior_W_sigma = prior_dict['conv2.W_sigma']
        self.conv2.prior_bias_mu = prior_dict['conv2.bias_mu']
        self.conv2.prior_bias_sigma = prior_dict['conv2.bias_sigma']
        # Modifying the prior，conv3
        self.conv3.prior_W_mu = prior_dict['conv3.W_mu']
        self.conv3.prior_W_sigma = prior_dict['conv3.W_sigma']
        self.conv3.prior_bias_mu = prior_dict['conv3.bias_mu']
        self.conv3.prior_bias_sigma = prior_dict['conv3.bias_sigma']
        # Modifying the prior，fc
        self.fc.prior_W_mu = prior_dict['fc.W_mu']
        self.fc.prior_W_sigma = prior_dict['fc.W_sigma']
        self.fc.prior_bias_mu = prior_dict['fc.bias_mu']
        self.fc.prior_bias_sigma = prior_dict['fc.bias_sigma']
        return None