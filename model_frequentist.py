import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random
import pandas as pd
import numpy as np

def get_model(name):
    # device is GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a model based on model name
    if name == "fcn_mnist":
        model = FCN_Mnist().to(device)
    elif name == "cnn_cifar":
        model = CNN_Cifar().to(device)
    elif name == "text_cnn":
        model = Text_CNN().to(device)
    else:
        print('model_name is wrong!')
    return model

# fcn_mnist model for Mnist
class FCN_Mnist(nn.Module):
    def __init__(self):
        super(FCN_Mnist, self).__init__()
        self.num_classes = 10
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# cnn_cifar model for Cifar10
class CNN_Cifar(nn.Module):
    def __init__(self):
        super(CNN_Cifar, self).__init__()
        self.num_classes = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=0, stride=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, stride=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, stride=1, bias=True)
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# text_cnn model for cnews
class Text_CNN(nn.Module):
    def __init__(self, vocab_size=50000, embedding_dim=16, num_filter=10,
                 filter_sizes=[2,3,4], output_dim=10):
        super().__init__()
        # embedding，embed a vocabulary of 50000 words into a 32-dimensional
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Convolutional layers, all num_filter (100) channels, volumes of length 2,3,4
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filter, kernel_size=(filter_sizes[0], embedding_dim))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=num_filter, kernel_size=(filter_sizes[1], embedding_dim))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=num_filter, kernel_size=(filter_sizes[2], embedding_dim))
        # 300->10
        self.fc = nn.Linear(len(filter_sizes) * num_filter, output_dim)
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
        return self.fc(torch.cat([maxpooled1,maxpooled2,maxpooled3], dim=1))