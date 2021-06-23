import numpy as np
import matplotlib.pyplot as plt

#%% Lessard function

def gradLessard(x):
    if x < 1:
        grad = 25*x
    elif x >= 1 and x < 2:
        grad = x + 24
    elif x >= 2:
        grad = 25*x - 24
        
    return grad

#%% Initializations and Parameters

add_noise = 0

if add_noise == 0:
    # For noiseless results in paper
    gamma = 2
    noise_std = 0
    num_iter = 20
else:
    # For noisy results in paper
    gamma  = 1.3
    noise_std = 0.1
    num_iter = 50

xd = 0

# In the range described by Lessard that will cause oscillations
xHB_inrange = 3.3
xHBm_inrange = 3.1

xHB0_inrange = xHB_inrange
xHBm0_inrange = xHBm_inrange

x_inrange = xHB_inrange
xm_inrange = xHBm_inrange

dyHBm_inrange = gradLessard(xHBm_inrange)

# Out of the oscillations range
xHB_outrange = 2
xHBm_outrange = 1

xHB0_outrange = xHB_outrange
xHBm0_outrange = xHBm_outrange

x_outrange = xHB_outrange
xm_outrange = xHBm_outrange

dyHBm_outrange = gradLessard(xHBm_outrange)

# True eigenvalues
LT = 25
lT = 1

# Optimal hyper-parameters
alfa = 4/((np.sqrt(LT) + np.sqrt(lT))**2)
beta = ((np.sqrt(LT) - np.sqrt(lT))/(np.sqrt(LT) + np.sqrt(lT)))**2

normxHB_inrange = np.zeros(num_iter)
normxHB_outrange = np.zeros(num_iter)
normx_inrange = np.zeros(num_iter)
normx_outrange = np.zeros(num_iter)

np.random.seed(4)

for k in range(num_iter):
    
    Noise = noise_std*np.random.randn(1)
    
    #%% In oscillations range
    
    dyHB_inrange = gradLessard(xHB_inrange) + Noise
    dxHB_inrange = xHB_inrange - xHBm_inrange
    
    dg_inrange = dyHB_inrange - dyHBm_inrange
    
    xHBm_inrange = xHB_inrange
    dyHBm_inrange = dyHB_inrange
    
    P_inrange = (dg_inrange*dg_inrange)/(dxHB_inrange*dxHB_inrange) 
    
    Lh_inrange = gamma*np.sqrt(P_inrange)
    
    xb_inrange = dg_inrange - Lh_inrange*dxHB_inrange
    
    p_inrange = (xb_inrange*xb_inrange)/(dxHB_inrange*dxHB_inrange)
    
    lh_inrange = np.sqrt(p_inrange)
    
    alfah_inrange = 4/((np.sqrt(Lh_inrange) + np.sqrt(lh_inrange))**2)
    betah_inrange = ((np.sqrt(Lh_inrange) - np.sqrt(lh_inrange))/(np.sqrt(Lh_inrange) + np.sqrt(lh_inrange)))**2
    
    normxHB_inrange[k] = np.linalg.norm(xHB_inrange - xd)
    
    xHB_inrange = xHB_inrange - alfah_inrange*dyHB_inrange + betah_inrange*dxHB_inrange
    
    #%% True optimal (in range comparison)
    
    dy_inrange = gradLessard(x_inrange) + Noise
    dx_inrange = x_inrange - xm_inrange
    
    xm_inrange = x_inrange
    
    normx_inrange[k] = np.linalg.norm(x_inrange - xd)
    
    x_inrange = x_inrange - alfa*dy_inrange + beta*dx_inrange
    
    dym_inrange = dy_inrange
    
    #%% Out of oscillations range
    
    dyHB_outrange = gradLessard(xHB_outrange) + Noise
    dxHB_outrange = xHB_outrange - xHBm_outrange
    
    dg_outrange = dyHB_outrange - dyHBm_outrange
    
    xHBm_outrange = xHB_outrange
    dyHBm_outrange = dyHB_outrange
    
    P_outrange = (dg_outrange*dg_outrange)/(dxHB_outrange*dxHB_outrange) 
    
    Lh_outrange = gamma*np.sqrt(P_outrange)
    
    xb_outrange = dg_outrange - Lh_outrange*dxHB_outrange
    
    p_outrange = (xb_outrange*xb_outrange)/(dxHB_outrange*dxHB_outrange)
    
    lh_outrange = np.sqrt(p_outrange)
    
    alfah_outrange = 4/((np.sqrt(Lh_outrange) + np.sqrt(lh_outrange))**2)
    betah_outrange = ((np.sqrt(Lh_outrange) - np.sqrt(lh_outrange))/(np.sqrt(Lh_outrange) + np.sqrt(lh_outrange)))**2
    
    normxHB_outrange[k] = np.linalg.norm(xHB_outrange - xd)
    
    xHB_outrange = xHB_outrange - alfah_outrange*dyHB_outrange + betah_outrange*dxHB_outrange
    
    #%% True optimal (out of range comparison)
    
    dy_outrange = gradLessard(x_outrange) + Noise
    dx_outrange = x_outrange - xm_outrange
    
    xm_outrange = x_outrange
    
    normx_outrange[k] = np.linalg.norm(x_outrange - xd)
    
    x_outrange = x_outrange - alfa*dy_outrange + beta*dx_outrange
    
    dym_outrange = dy_outrange
    
#%% Optimal convergence rate

conv_rate_opt = ((np.sqrt(LT) - np.sqrt(lT))/(np.sqrt(LT) + np.sqrt(lT)))

kk = range(num_iter)

conv_rate_inrange = np.linalg.norm(xHBm0_inrange-xd)*conv_rate_opt**kk
conv_rate_outrange = np.linalg.norm(xHBm0_outrange-xd)*conv_rate_opt**kk

#%% Plots

if add_noise == 0:
    fig, ax = plt.subplots(2, sharex=True)
    
    ax[0].semilogy(normxHB_inrange)
    ax[0].semilogy(normx_inrange)
    ax[0].semilogy(conv_rate_inrange)
    # # ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('$|x_k - x^*|$')
    ax[0].set_title('$x_1$ = '+str(xHB0_inrange)+' and $x_0$ = '+str(xHBm0_inrange))
    ax[0].legend(['Proposed HB', 'Optimal HB', 'Optimal Convergence Rate'])
    ax[0].grid()
    
    ax[1].semilogy(normxHB_outrange)
    ax[1].semilogy(normx_outrange)
    ax[1].semilogy(conv_rate_outrange)
    ax[1].set_xlabel('Iterations, $k$')
    ax[1].set_ylabel('$|x_k - x^*|$')
    ax[1].set_title('$x_1$ = '+str(xHB0_outrange)+' and $x_0$ = '+str(xHBm0_outrange))
    ax[1].legend(['Proposed HB', 'Optimal HB', 'Optimal Convergence Rate'])
    ax[1].grid()
    
    plt.xticks(range(num_iter),range(1,num_iter+1))
else:
    plt.figure()
    plt.semilogy(normxHB_outrange)
    plt.semilogy(normx_outrange)
    # plt.semilogy(conv_rate_outrange)
    plt.xlabel('Iterations, $k$', fontsize=12)
    plt.ylabel('$|x_k - x^*|$', fontsize=12)
    plt.title('$x_1$ = '+str(xHB0_outrange)+' and $x_0$ = '+str(xHBm0_outrange))
    plt.legend(['Proposed HB', 'Optimal HB'], fontsize=12)
    # plt.legend(['Proposed SHB', 'Optimal HB', 'Optimal Convergence Rate'])
    plt.grid()
    plt.xticks([0,9,19,29,39,49],[1,10,20,30,40,50])

