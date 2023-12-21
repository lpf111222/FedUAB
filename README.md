# FedUAB
we integrate Bayesian neural network (BNN) into Federated Learning(FL) framework, a distributed variational inference method has been implemented. 
In this hybrid approach, each FL client independently trains a BNN using the Bayes by Backprop algorithm, the model weights are approximated as Gaussian distributions, which mitigate overfitting issues and ensure data privacy.
we present a novel method to overcome several key challenges in the fusion of BNN and FL, these challenges include selecting an optimal prior distribution, aggregating weights characterized by Gaussian forms across multiple clients, and rigorously managing weights variances.
We named this algorithm as FedUAB (Uncertainty-Aware BNN for FL).


The code will be released as soon as the paper is accepted.
