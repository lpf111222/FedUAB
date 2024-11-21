from torchvision import datasets, transforms
import numpy as np
import random

def get_dataset(dir, conf):
    if conf["dataset_name"] == 'mnist':
        # train data, train=True
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        # eval data, train=False
        eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
    elif conf["dataset_name"] == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # train data, train=True
        train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        # eval data, train=False
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
    else:
        print("Wrong dataset name.")
        # sample training data amongst users
    if conf["data_distribution"] == 'IID':
        clients_distribution, train_clients_index, eval_clients_index = client_iid(train_dataset, eval_dataset, conf)
    elif conf["data_distribution"] == 'Non-IID-2':
        clients_distribution, train_clients_index, eval_clients_index = client_noniid_two_label(train_dataset, eval_dataset, conf)
    elif conf["data_distribution"] == 'Dirichlet':
        clients_distribution, train_clients_index, eval_clients_index = client_noniid_Dirichlet(train_dataset, eval_dataset, conf)
    else:
        print("Client data distribution error，The corrects  are：IID、Non-IID-2、Dirichlet.")
    return train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index


# The samples in the dataset are separated by label
def get_every_lable_list(dataset):
    # training and test sets, we separate the list by label and split it into lists of labels
    num_class = len(dataset.classes)
    # Holds the index values for 10 categories, and the elements in the list are lists
    every_lable_list = []
    for _ in range(num_class):
        every_lable_list.append([])
    # lable
    lable_list = list(dataset.class_to_idx.values())
    for i in range(len(dataset.targets)):
        lable_index = lable_list.index(dataset.targets[i])
        every_lable_list[lable_index].append(i)
    # shuffle
    for l in every_lable_list:
        random.shuffle(l)
    return every_lable_list

# Assign data to each client according to the data distribution
def fill_data_with_distribution(dataset, clients_distribution, conf):
    # Number of classes in the dataset
    num_class = len(dataset.classes)
    # Number of samples per client
    # When the num_small_data hyperparameter is 0, the training data is split equally among all clients;
    # When the num_small_data hyperparameter is a specific number, the data is directly the number of client samples,
    # and the small data scenario in the paper is 50
    if conf["num_small_data"] == 0 or dataset.train == False:
        num_client_data = int(len(dataset.targets) / conf["num_client"])
    else:
        num_client_data = conf["num_small_data"]
    # The samples in the dataset are separated by label
    every_lable_list = get_every_lable_list(dataset)
    clients_index = {}
    for i in range(conf["num_client"]):
        clients_index[i] = []
        distribution_data = (clients_distribution[i] * num_client_data).astype('int')
        # Deal with small problems that are not divisible
        distribution_data[np.random.randint(low=0, high=num_class)] += (num_client_data - np.sum(distribution_data))
        # Add samples to each class one by one
        for j in range(num_class):
            num_samples = distribution_data[j]
            # Avoid some special case bugs, rarely used.
            # Dirichlet distribution, there are some extreme cases where you don't have enough samples for some classes.
            if len(every_lable_list[j]) < num_samples:
                every_lable_list[j] += get_every_lable_list(dataset)[j]
            # add samples
            for k in range(num_samples):
                clients_index[i].append(every_lable_list[j].pop())
    return clients_index


def client_iid(train_dataset, eval_dataset, conf):
    num_class = len(train_dataset.classes)
    clients_distribution = {}
    # iid is the label uniform distribution
    for i in range(conf["num_client"]):
        clients_distribution[i] = np.ones(num_class) / num_class
    train_clients_index = fill_data_with_distribution(train_dataset, clients_distribution, conf)
    eval_clients_index = fill_data_with_distribution(eval_dataset, clients_distribution, conf)
    return clients_distribution, train_clients_index, eval_clients_index


def client_noniid_two_label(train_dataset, eval_dataset, conf):
    num_class = len(train_dataset.classes)
    clients_distribution = {}
    # Each client has a random two-class label
    list_classes = list(range(num_class))
    random.shuffle(list_classes)
    for i in range(conf["num_client"]):
        clients_distribution[i] = np.zeros(num_class)
        # noniid_two_label，each client has a random two-class label
        if len(list_classes) < 2:
            list_classes = list(range(num_class))
            random.shuffle(list_classes)
        clients_distribution[i][list_classes.pop()] = 0.5
        clients_distribution[i][list_classes.pop()] = 0.5
    train_clients_index = fill_data_with_distribution(train_dataset, clients_distribution, conf)
    eval_clients_index = fill_data_with_distribution(eval_dataset, clients_distribution, conf)
    return clients_distribution, train_clients_index, eval_clients_index


def client_noniid_Dirichlet(train_dataset, eval_dataset, conf):
    num_class = len(train_dataset.classes)
    clients_distribution = {}
    # Dirichlet distribution alpha parameter, uniformly distributed over num_clients [dirichlet_alpha_min,dirichlet_alpha_max]
    alpha_list = []
    for i in range(conf["num_client"]):
        alpha_list.append(
            ((conf["dirichlet_alpha_max"] - conf["dirichlet_alpha_min"]) / conf["num_client"]) * (i + 1) + conf[
                "dirichlet_alpha_min"])
    # The distribution of each client is sampled following the alpha parameter of the Dirichlet distribution
    for i in range(conf["num_client"]):
        clients_distribution[i] = np.random.dirichlet((np.ones(num_class) * alpha_list[i]), 1)[0]
    train_clients_index = fill_data_with_distribution(train_dataset, clients_distribution, conf)
    eval_clients_index = fill_data_with_distribution(eval_dataset, clients_distribution, conf)
    return clients_distribution, train_clients_index, eval_clients_index