This repository contains implementations for the Adaptive Heavy-Ball optimizer proposed in "An Adaptive Polyak Heavy-Ball Method with Reduced Tuning" in Python (Pytorch for its implementation in deep neural networks).

The developed adaptive heavy-ball (AHB) assumes that the objective function can be represented with time-varying quadratic functions and estimates the Polyak's optimal hyper-parameters at each iteration. AHB employs the absolute differences of current and previous model parameters and their gradients. Such representation allows for a computationally efficient optimizer. 

AHB attains global convergence for convex objective functions with Lipschitz-continuous gradients. AHB also guarantees a global linear convergence rate for strictly convex objective functions. The effectiveness and superiority of AHB is validated on non-convex functions and image classification datasets, while, at worst, requiring the tuning of only one hyper-parameter.

# Experiments

The convergence rate, overall performance, and robustness of AHB is studied.

To demonstrate the reduced tuning efforts that AHB offers, AHB is only tuned where it is reasonable to do so. To that end, AHB is tuned on one-dimensional (Lessard) and two-dimensional (Beale) functions, where tuning is cheap. The effect of tuning is also showed on an example including quadratic cost functions. In large-scale systems however, tuning is computationally expensive, and thus AHB is implemented on deep neural networks with no tuning involved, by setting gamma=1.

**Lessard Problem**

The Lessard problem in [1] uses a strongly convex function, with non twice differentiable gradients. The superiority of the convergence rate of AHB when compared to the optimal heavy-ball (HB) method [2] is shown on the Lessard problem.

![Lessard_Example](https://user-images.githubusercontent.com/44982976/123044656-f78b8280-d3c7-11eb-8de7-f078039ad073.png)

The figue above is a visualization of the convergence rate of AHB versus optimal HB via the norm of the error at every iteration (k).

![Lessard_Example3_v2](https://user-images.githubusercontent.com/44982976/123048772-c5305400-d3cc-11eb-8239-86c37e699f3d.png)

The figure above displayes norm of the error produced by AHB veresus the optimal HB in the presence of noisy gradients.

**Positive Semi-Definite Quadratic Functions**

The AHB is compared with a time-invariant and time-varying HB on positive semi-definite quadratic functions.



**Beale Function**

The convergence rate of AHB is evaluated by comparing it with first-order optimizers on the (non-convex and non-quadratic) Beale function.

AHB is compared with stochastic gradient descent (SGD), SGD with Momentum (SGDm) [3], Nesterov's accelerated gradient (NAG) method [4], AdaGrad [5], RMSProp [6], and Adam [7]. The learning rates for all optimizers, excluding MADAGRAD, are chosen by conducting a random search over the values {0.5, 0.1, 0.05, 0.01, 0.005, 0.001}, where each optimizer is run over 1000 random initializations of the model parameters. The optimizer parameters that returned the fastest convergence rates were chosen. 

The learning rates that are selected for SGD, SGDm, NAG, RMSProp, Adagrad, and Adam are 0.01, 0.01, 0.001, 0.01, 0.5, and 0.5, respectively. The momentum factor used for SGDm is the standard value of 0.9, and the standard values of beta_1=0.9 and beta_2=0.99 are used for Adam. As for oAHB, a hyper-parameter search for gamma was conducted, sweeping over the values {1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2}. The value taken for all hyper-parameter searches was the one that returned the largest number of times each optimizer converged. The value for gamma for AHB was chosen to be 1.2. 

The performance metrics considered are the number of times each optimizer converges, the average number of steps takent to converge (given converged), and wins. A win is recorded for the optimizers that converge to a solution first within the first 1000 iterations.

The table below shows the convergence rate of AHB in comparison to known optimizers when tested on the Beale function.

Optimizer | Wins | Times converged | Average steps 
--- | --- | --- | ---
AHB | 801 | 915 | 175.52 
SGD | 0 | 27 | 992.94  
SGDm | 32 | 319 | 719.11 
NAG | 12 | 569 | 575.23 
RMSProp | 2 | 86 | 933.54 
Adagrad | 13 | 346 | 772.11 
Adam | 76 | 535 | 586.37 

**Image Classification**

AHB's performance is evaluated against popular optimizers on MNIST, QMNIST, CIFAR-10, and CIFAR-100, where the lower-bound of the adaptive learning rate is fixed to 0.05 as default value for image classification.

The MNIST dataset is a 10-class image classification dataset composed of 60,000 and 10,000 training and testing grey-scale images of hand-written digits, respectively. QMNIST extends MNIST's testing set to 60,000 testing images. The CIFAR-10 and CIFAR-100 datasets consist of 50,000 training images and 10,000 testing images with dimensions 32x32 with 10 and 100 classes, respectively.

For MNIST and QMNIST, the neural network used is the conventional convolutional neural network (CNN) as designed in [8], which includes two convolutional layers with kernel size 5, one fully-connected hidden layer, and a proceeding fully-connected classification layer of 50 neurons. The activation function chosen is the ReLU function. The networks are run over 5 random (seeds) initializations of the network parameters, and a batch size of 64 is used.

The learning rate chosen for diffGrad is 0.001 with the standard values of beta_1=0.9 and beta_2=0.99. For SGDm, a learning rate value of 0.01 is chosen with beta = 0.9. For Adam, a learning rate of 0.0005 is chosen with beta_1=0.9 and beta_2=0.99. For NAG, a learning rate of 0.001 is chosen and beta=0.9. Lastly, for AdamW, a learning rate of 0.0005 is chosen with beta_1=0.9 and beta_2=0.99, and a weight decay value of 1.

# How to Execute Codes


# References

[1] Lessard, Laurent, Benjamin Recht, and Andrew Packard. "Analysis and design of optimization algorithms via integral quadratic constraints." SIAM Journal on Optimization 26.1 (2016): 57-95.

[2] Polyak, Boris T. "Introduction to optimization. optimization software." Inc., Publications Division, New York 1 (1987).

[3] Qian, Ning. "On the momentum term in gradient descent learning algorithms." Neural networks 12.1 (1999): 145-151.

[4] Nesterov, Yu. "A method of solving a convex programming problem with convergence rate O (1/k^ 2) O (1/k2)." Sov. Math. Dokl. Vol. 27. No. 2.

[5] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods for online learning and stochastic optimization." Journal of machine learning research 12.7 (2011).

[6] Hinton, Geoffrey, Nitish Srivastava, and Kevin Swersky. "Neural networks for machine learning lecture 6a overview of mini-batch gradient descent." Cited on 14.8 (2012).

[7] Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).

[8] Gregor Koehler, "Mnist handwritten digit recognition in pytorch." url=https://nextjournal.com/gkoehler/pytorch-mnist, (2020).

[9] Zhang, Michael R., et al. "Lookahead optimizer: k steps forward, 1 step back." arXiv preprint arXiv:1907.08610 (2019).

[10] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
