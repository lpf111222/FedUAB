# FedUAB
Source-code of paper ‘Federated Learning meets Bayesian Neural Network: Robust and Uncertainty-Aware Distributed Variational Inference', the paper is under review.
This papre integrate Bayesian neural network (BNN) into Federated Learning(FL) framework, a distributed variational inference method has been implemented. 
In this hybrid approach, each FL client independently trains a BNN using the Bayes by Backprop algorithm, the model weights are approximated as Gaussian distributions, which mitigate overfitting issues and ensure data privacy.
we present a novel method to overcome several key challenges in the fusion of BNN and FL, these challenges include selecting an optimal prior distribution, aggregating weights characterized by Gaussian forms across multiple clients, and rigorously managing weights variances.
We named this algorithm as FedUAB (Uncertainty-Aware BNN for FL).

To run the code, run the command:   python main.py -c ./conf.json

All hyperparameters are written to file conf.json, which can be rewritten to run different Settings, Let's briefly explain what each hyperparameter means:
(1) 
