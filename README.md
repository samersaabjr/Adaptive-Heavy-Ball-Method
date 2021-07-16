This repository contains implementations for the Adaptive Heavy-Ball optimizer proposed in "An Adaptive Heavy-Ball Method" in Python (Pytorch for its implementation in deep neural networks).

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

The AHB is compared with a time-invariant and time-varying HB (satisfying the conditions of the work in [3]) on positive semi-definite quadratic functions. The quadratic objective functions with positive semi-definite quadratic dxd matrices are considered for dimensions d = {2,...100}. The initial guess is sampled from a normal distribution, N(0,1). For each dimension, 20d iterations are run.

![Quadratics_1](https://user-images.githubusercontent.com/44982976/126006319-dac00d7a-f46e-40d7-9f45-f72e6daa26d6.jpg)

The figure above shows the comparison of the progress of the objective values with A for d = 50 evaluated at the Cesaro average of the iterates of the three heavy-ball methods under study.

![Quadratics_2](https://user-images.githubusercontent.com/44982976/126006328-a3e6aa01-e768-4cca-84fc-77f60df9fab9.jpg)

The figure above shows the comparison of the objective values with A for d = 2, ..., 100 evaluated at the Cesaro average at the iterate k = 20d of the three heavy-ball methods under study.

![Quadratics_3](https://user-images.githubusercontent.com/44982976/126006340-95527d4c-bbc1-4add-ba9d-59d05fa2cce6.png)

The figure above shows the comparison of the objective values with A for d = 50 evaluated at the Cesaro average at the iterate k = 1,000 of the proposed adaptive heavy-ball method for different values of gamma.

**Beale Function**

The convergence rate of AHB is evaluated by comparing it with first-order optimizers on the (non-convex and non-quadratic) Beale function.

AHB is compared with stochastic gradient descent (SGD), SGD with Momentum (SGDm) [4], Nesterov's accelerated gradient (NAG) method [5], AdaGrad [6], RMSProp [7], and Adam [8]. The learning rates for all optimizers, excluding AHB, are chosen by conducting a random search over the values {0.5, 0.1, 0.05, 0.01, 0.005, 0.001}, where each optimizer is run over 1000 random initializations of the model parameters. The optimizer parameters that returned the fastest convergence rates were chosen. 

The learning rates that are selected for SGD, SGDm, NAG, RMSProp, Adagrad, and Adam are 0.01, 0.01, 0.001, 0.01, 0.5, and 0.5, respectively. The momentum factor used for SGDm is the standard value of 0.9, and the standard values of beta_1=0.9 and beta_2=0.99 are used for Adam. As for AHB, a hyper-parameter search for gamma was conducted, sweeping over the values {1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2}. The value taken for all hyper-parameter searches was the one that returned the largest number of times each optimizer converged. The value for gamma for AHB was chosen to be 1.2. 

The performance metrics considered are the number of times each optimizer converges, the average number of steps takent to converge (given converged), and wins. A win is recorded for the optimizers that converge to a solution first within the first 1000 iterations.

The table below shows the convergence rate of AHB in comparison to known optimizers when tested on the Beale function.

Optimizer | Wins | Times converged | Average steps 
--- | --- | --- | ---
AHB | 801 | 915 | 175.52 
SGD | 0 | 27 | 992.94  
SGDm | 34 | 319 | 719.11 
NAG | 12 | 569 | 575.23 
RMSProp | 3 | 86 | 933.54 
Adagrad | 9 | 346 | 772.11 
Adam | 69 | 535 | 586.37 

**Image Classification**

AHB's performance is evaluated against popular optimizers on MNIST, QMNIST, CIFAR-10, and CIFAR-100, where the lower-bound of the adaptive learning rate is fixed to 0.05 as default value for image classification. Specifically, AHB is compared with SGDm [4], NAG [5], Adam [8], and AdamW [9].

The MNIST dataset is a 10-class image classification dataset composed of 60,000 and 10,000 training and testing grey-scale images of hand-written digits, respectively. QMNIST extends MNIST's testing set to 60,000 testing images. The CIFAR-10 and CIFAR-100 datasets consist of 50,000 training images and 10,000 testing images with dimensions 32x32 with 10 and 100 classes, respectively.

For MNIST and QMNIST, the neural network used is the conventional convolutional neural network (CNN) as designed in [10], which includes two convolutional layers with kernel size 5, one fully-connected hidden layer, and a proceeding fully-connected classification layer of 50 neurons. The activation function chosen is the ReLU function. The networks are run over 5 random (seeds) initializations of the network parameters, and a batch size of 64 is used.

For SGDm, a learning rate value of 0.01 is chosen with beta = 0.9. For Adam, a learning rate of 0.0005 is chosen with beta_1=0.9 and beta_2=0.99. For NAG, a learning rate of 0.01 is chosen and beta=0.9. Lastly, for AdamW, a learning rate of 0.0005 is chosen with beta_1=0.9 and beta_2=0.99, and a weight decay value of 1.

For the CIFAR experiments, the tuning parameters set forth by Zhang et al. [11] are adopted for SGDm and AdamW, and similarly run the experiments using a Resnet-18 [12] for three different seeds using a batch size of 128. Additionally, Adam is tuned in-house. All optimizers are tuned on CIFAR-10, then the same hyper-parameters are used on CIFAR-100.

SGDm uses beta=0.9, learning rate of 0.05, and weight decay value of 0.001. AdamW has a learning rate of 0.0003 and weight decay value of 1. For NAG, a learning rate of 0.05 is used and beta=0.9. For Adam, a learning rate value of 0.005 is chosen with beta_1=0.9 and beta_2=0.99.

![cifar_acc_results](https://user-images.githubusercontent.com/44982976/123177294-fd7a7580-d452-11eb-8527-97d302024e85.png)

The figure above shows the test set accuracy at every epoch for all optimizers on both CIFAR-10 (top) and CIFAR-100 (bottom). The solid lines reflect the mean accuracy over the three random runs, whereas the shaded regions reflect the corresponding standard deviations.

# How to Execute Codes

**Lessard Problem**

These experiments are run using the Numpy library provided in Python. The codes are in the "Lessard" folder. To run the entire experiment, simply run "main_Lessard.py". To reproduce the results above, just select whether or not you want noisy gradients by setting "add_noise" to 1 or 0. The gamma and number of iterations ("num_iter") are set to repreoduce the same results as above, however the variables can be changed as desired. The variables ending with "_inrange" correspond to the experiment where the initial conditions are unfiormly sampled from the range -5 <= x0 < 1, and the variables ending with "_outrange" correspond to the experiment where the initial conditions are unfiormly sampled from the range 1 <= x0 <= 5.

**Positive Semi-Definite Quadratic Functions**

This is the only experiment written in MATLAB. Simply run the script "PosSemiDef_Quadratic.m". The code is split into two sections. The first section will plot the cesaro average of iterates for a gamma of the user's choice (just populate gamma variable) for the case of dimension d = 50, as well as the cesaro average at the last iterate for all dimensions. The second section plots the different cesaro average at the last iterate for different values of gamma for the dimension d = 50 case.

**Beale Function**

These experiments are run using the Numpy library provided in Python. The codes are in the "Beale" folder. The code for each optimizer is titled "OptimizerName_Class_Numpy.py". To run the experiment, simply run "MAIN_Beale.py". The number of runs and iterations are set to 1000 each, but can be modified if desired. When the code is run it will save text files titled as the optimizer name which saves the history of the parameter updates as they reach the solution for the last run.

**Image Classification**

The codes are in the "Image Classification" folder. The AHB optimizer itself can be found in the file "AHB.py". To run the AHB optimizer on any of the image classification datasets, simply run "main_MNIST.py", "main_QMNIST.py", "main_cifar10.py", or "main_cifar100.py". You will need to specify the optimizer you are using by populating the "method" variable. The choices are 'ahb', 'sgd', 'nag', 'adam', or 'adamw' (e.g. method = 'ahb').

# References

[1] Lessard, Laurent, Benjamin Recht, and Andrew Packard. "Analysis and design of optimization algorithms via integral quadratic constraints." SIAM Journal on Optimization 26.1 (2016): 57-95.

[2] Polyak, Boris T. "Introduction to optimization. optimization software." Inc., Publications Division, New York 1 (1987).

[3] Ghadimi, Euhanna, Hamid Reza Feyzmahdavian, and Mikael Johansson. "Global convergence of the heavy-ball method for convex optimization." 2015 European control conference (ECC). IEEE, 2015

[4] Qian, Ning. "On the momentum term in gradient descent learning algorithms." Neural networks 12.1 (1999): 145-151.

[5] Nesterov, Yu. "A method of solving a convex programming problem with convergence rate O (1/k^ 2) O (1/k2)." Sov. Math. Dokl. Vol. 27. No. 2.

[6] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods for online learning and stochastic optimization." Journal of machine learning research 12.7 (2011).

[7] Hinton, Geoffrey, Nitish Srivastava, and Kevin Swersky. "Neural networks for machine learning lecture 6a overview of mini-batch gradient descent." Cited on 14.8 (2012).

[8] Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).

[9] Loshchilov, Ilya, and Frank Hutter. "Fixing weight decay regularization in adam." (2018).

[10] Gregor Koehler, "Mnist handwritten digit recognition in pytorch." url=https://nextjournal.com/gkoehler/pytorch-mnist, (2020).

[11] Zhang, Michael R., et al. "Lookahead optimizer: k steps forward, 1 step back." arXiv preprint arXiv:1907.08610 (2019).

[12] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
