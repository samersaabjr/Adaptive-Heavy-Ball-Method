from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

from Adam_Class_Numpy import AdamOptimizer
from Adagrad_Class_Numpy import AdagradOptimizer
from RMSProp_Class_Numpy import RMSPropOptimizer
from Momentum_Class_Numpy import MomentumOptimizer
from SGD_Class_Numpy import GradientDescentOptimizer
from NAG_Class_Numpy import NAGOptimizer
from AHB_Class_Numpy import AHBOptimizer

#%% Beale Function

f = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def gradients(x, y):
  """Gradient of Beale function.

  Args:
    x: x-dimension of inputs
    y: y-dimension of inputs

  Returns:
    grads: [dx, dy], shape: 1-rank Tensor (vector) np.array
      dx: gradient of Beale function with respect to x-dimension of inputs
      dy: gradient of Beale function with respect to y-dimension of inputs
  """
  dx = 2. * ( (1.5 - x + x * y) * (y - 1) + \
                (2.25 - x + x * y**2) * (y**2 - 1) + \
                (2.625 - x + x * y**3) * (y**3 - 1) )
  dy = 2. * ( (1.5 - x + x * y) * x + \
              (2.25 - x + x * y**2) * 2. * x * y + \
              (2.625 - x + x * y**3) * 3. * x * y**2 )
  grads = np.array([dx, dy])
  return grads


minima = np.array([3., .5])
minima_ = minima.reshape(-1, 1)
print("minima (1x2 row vector shape): {}".format(minima))
print("minima (2x1 column vector shape):")
print(minima_)

#%% putting together our points to plot in a 3D plot

number_of_points = 50
margin = 4.5
x_min = 0. - margin
x_max = 0. + margin
y_min = 0. - margin
y_max = 0. + margin
x_points = np.linspace(x_min, x_max, number_of_points)
y_points = np.linspace(y_min, y_max, number_of_points)
x_mesh, y_mesh = np.meshgrid(x_points, y_points)
z = np.array([f(xps, yps) for xps, yps in zip(x_mesh, y_mesh)])

#%% 3D Plot with Minima

# fig = plt.figure(figsize=(10, 8))
# ax = plt.axes(projection='3d', elev=80, azim=-100)

# ax.plot_surface(x_mesh, y_mesh, z, norm=LogNorm(), rstride=1, cstride=1,
#                 edgecolor='none', alpha=.8, cmap=plt.cm.jet)
# ax.plot(*minima_, f(*minima_), 'r*', markersize=20)

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# ax.set_xlim((x_min, x_max))
# ax.set_ylim((y_min, y_max))

# #plt.draw()
# plt.show()

#%% Train Optimizer

num_runs = 1000
num_epochs = 1000
num_opts = 7

stdev = 4

steps_tracker = np.zeros((num_opts,num_runs))
final_sol = np.zeros((num_opts,num_runs))

for i in range(num_runs):
    
    np.random.seed(i)
    
    x0 = stdev*np.random.rand(1)
    y0 = stdev*np.random.rand(1)
    
    opt_ahb = AHBOptimizer(f, gradients, x_init=x0, y_init=y0, c=1.2)
    opt_ahb.train(num_epochs)
    steps_tracker[0,i] = len(opt_ahb.z_history)
    final_sol[0,i] = opt_ahb.z_history[-1:][0]
    
    opt_SGD = GradientDescentOptimizer(f, gradients, x_init=x0, y_init=y0, learning_rate=0.01)
    opt_SGD.train(num_epochs)
    steps_tracker[1,i] = len(opt_SGD.z_history)
    final_sol[1,i] = opt_SGD.z_history[-1:][0]
    
    opt_momentum = MomentumOptimizer(f, gradients, x_init=x0, y_init=y0, learning_rate=0.01, momentum=0.9)
    opt_momentum.train(num_epochs)
    steps_tracker[2,i] = len(opt_momentum.z_history)
    final_sol[2,i] = opt_momentum.z_history[-1:][0]
    
    opt_RMSProp = RMSPropOptimizer(f, gradients, x_init=x0, y_init=y0, learning_rate=0.01, decay=0.9, epsilon=1e-10)
    opt_RMSProp.train(num_epochs)
    steps_tracker[3,i] = len(opt_RMSProp.z_history)
    final_sol[3,i] = opt_RMSProp.z_history[-1:][0]
    
    opt_adagrad = AdagradOptimizer(f, gradients, x_init=x0, y_init=y0, learning_rate=0.5)
    opt_adagrad.train(num_epochs)
    steps_tracker[4,i] = len(opt_adagrad.z_history)
    final_sol[4,i] = opt_adagrad.z_history[-1:][0]
    
    opt_adam = AdamOptimizer(f, gradients, x_init=x0, y_init=y0, learning_rate=0.5, beta1=0.9, beta2=0.999, epsilon=1e-8)
    opt_adam.train(num_epochs)
    steps_tracker[5,i] = len(opt_adam.z_history)
    final_sol[5,i] = opt_adam.z_history[-1:][0]
    
    opt_nag = NAGOptimizer(f, gradients, x_init=x0, y_init=y0, learning_rate=0.001, momentum=0.9)
    opt_nag.train(num_epochs)
    steps_tracker[6,i] = len(opt_nag.z_history)
    final_sol[6,i] = opt_nag.z_history[-1:][0]

min_steps_recorder = np.zeros(num_opts)

for i in range(num_runs):
    
    if np.sum(steps_tracker[:,i]==np.min(steps_tracker[:,i]))==1:
        j = np.argmin(steps_tracker[:,i])
        min_steps_recorder[j] += 1

avg_steps_all = np.mean(steps_tracker,axis=1)

for i in range(len(steps_tracker)):
    print(np.sum(steps_tracker[i,:]<num_epochs))

#%% Results

print('Number of wins for AHB: '+str(min_steps_recorder[0]))
print('Number of wins for SGD: '+str(min_steps_recorder[1]))
print('Number of wins for SGDm: '+str(min_steps_recorder[2]))
print('Number of wins for NAG: '+str(min_steps_recorder[6]))
print('Number of wins for RMSProp: '+str(min_steps_recorder[3]))
print('Number of wins for AdaGrad: '+str(min_steps_recorder[4]))
print('Number of wins for Adam: '+str(min_steps_recorder[5]))

print('\n')

print('Average nmber of steps to converge for AHB: '+str(avg_steps_all[0]))
print('Average nmber of steps to converge for SGD: '+str(avg_steps_all[1]))
print('Average nmber of steps to converge for SGDm: '+str(avg_steps_all[2]))
print('Average nmber of steps to converge for NAG: '+str(avg_steps_all[6]))
print('Average nmber of steps to converge for RMSProp: '+str(avg_steps_all[3]))
print('Average nmber of steps to converge for AdaGrad: '+str(avg_steps_all[4]))
print('Average nmber of steps to converge for Adam: '+str(avg_steps_all[5]))

print('\n')

print('Number of times AHB converged: '+str(np.sum(steps_tracker[0,:]<num_epochs)))
print('Number of times SGD converged: '+str(np.sum(steps_tracker[1,:]<num_epochs)))
print('Number of times SGDm converged: '+str(np.sum(steps_tracker[2,:]<num_epochs)))
print('Number of times NAG converged: '+str(np.sum(steps_tracker[6,:]<num_epochs)))
print('Number of times RMSProp converged: '+str(np.sum(steps_tracker[3,:]<num_epochs)))
print('Number of times AdaGrad converged: '+str(np.sum(steps_tracker[4,:]<num_epochs)))
print('Number of times Adam converged: '+str(np.sum(steps_tracker[5,:]<num_epochs)))
