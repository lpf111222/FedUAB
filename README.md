# FedUAB
Source-code of paper ‘Federated Learning meets Bayesian Neural Network: Robust and Uncertainty-Aware Distributed Variational Inference', the paper is under review.
This papre integrate Bayesian neural network (BNN) into Federated Learning(FL) framework, a distributed variational inference method has been implemented. 
In this hybrid approach, each FL client independently trains a BNN using the Bayes by Backprop algorithm, the model weights are approximated as Gaussian distributions, which mitigate overfitting issues and ensure data privacy.
we present a novel method to overcome several key challenges in the fusion of BNN and FL, these challenges include selecting an optimal prior distribution, aggregating weights characterized by Gaussian forms across multiple clients, and rigorously managing weights variances.
We named this algorithm as FedUAB (Uncertainty-Aware BNN for FL).

To run the code, run the command:   python main.py -c ./conf.json
All hyperparameters are written to file conf.json, which can be rewritten to run different Settings.These hyperparameters are all detailed in the paper, Let's briefly explain what each hyperparameter means: 
(1) model_name："fcn_mnist" represents the fully connected network for the mnist dataset, "cnn_cifar" represents the convolutional network for the cifar10 dataset;
(2) dataset_name： "mnist" for the mnist dataset, "cifar10" for the cifar-10 dataset;
(3) num_client：the total number of FL clients, the experiment is 100, can be modified to other positive integers;
(4) k：the number of participating clients in each round of FL training, which is 10 in the experiment of the paper and can be modified to an integer less than or equal to num_client;
(5) global_epochs：total number of FL training sessions (2000 for the paper's experiments)
(6)batch_size：the batch_size of the client training, the experiment of the paper is 10;
(7)local_epochs：the total number of client-side local data training epochs for each client training, the experiment of the paper is 5;
(8)lr : the learning rate for client-side training; 0.05 in experiment of the paper;
(9)kl_division: the coefficient of the complexity loss calculated by the KL divergence, which is 0.0001 in our experiment. See parameter γ in Section 4.4 of the paper for details;
(10)data_distribution： "IID" means that the client data is independent and identically distributed, "Dirichlet" means that the client data is Dirichlet distributed to sample, and "Non-IID-2" means that there are only two data labels for each client. See Section 5.3 Experimental Setting for details;
(11)dirichlet_alpha_min，dirichlet_alpha_max：hyperparameters for Dirichlet distribution of client data that adjust the degree of NonIID. See Section 5.3 Experimental Setting for details.  It is 0-0.5 in our paper;
(12)num_small_data: When the num_small_data hyperparameter is 0, the training data is split equally among all clients;  When the num_small_data hyperparameter is a specific number, the data is directly the number of client samples, and the small data scenario in the paper is 50.

In main.py, it defines FedUAB and other baselines for training and testing. You can add or delete it yourself. Each Baseline defines a class in a separate file(such as FL_FedUAB.py), which inherits the base classes ServerBase(server_base.py) and ClientBase(client_base.py) of FL. The newly defined class only rewrites the innovation point that it is different from the classical FL, and very easy to extend with new algorithmic code.
