import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class AHB(Optimizer):
    r"""Implements the Adaptive Polyak Heavy-Ball (AHB) algorithm,
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        gamma (float): scales the approximation of the Lipschitz constant L (default at 1)
        epsilon (float): placed in denominators to avoid singularities (default at 5e-3)
        weight_decay (float): weight decay value (default at 0)
        
    Example:
        >>> from AHB import AHB
        >>> optimizer = AHB(model.parameters())
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, gamma=1, epsilon=0.005, dampening=0, weight_decay=0):
        
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(gamma=gamma, epsilon=epsilon, dampening=dampening, weight_decay=weight_decay)
        
        super(AHB, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AHB, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            gamma = group['gamma']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                
                state = self.state[p]
                
                p = p.flatten()
                d_p = d_p.flatten()
                
                p = torch.reshape(p,(p.size()[0],1))
                d_p = torch.reshape(d_p,(d_p.size()[0],1))
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['prev_p'] = torch.zeros_like(p)
                    state['d_p0'] = torch.ones_like(d_p)
                    state['Lh'] = 1
                    state['lh'] = 1
                
                prev_p, d_p0, Lh, lh = state['prev_p'], state['d_p0'], state['Lh'], state['lh']
                
                state['step'] += 1
                
                dx = torch.abs(p - prev_p)
                
                # max eig
                P = torch.div(torch.matmul((d_p - d_p0).T,(d_p - d_p0)),torch.matmul(dx.T,dx)+epsilon)
                
                Lh = gamma*torch.sqrt(P)
                
                # min eig
                xb = d_p - d_p0 - Lh*dx
                
                pp = torch.div(torch.matmul(xb.T,xb),torch.matmul(dx.T,dx)+epsilon)
                
                lh =  torch.sqrt(pp)

                
                d_p0.mul_(0).add_(d_p)
                
                if lh > Lh:
                    lh_temp = lh
                    lh = Lh
                    Lh = lh_temp
                
                alpha_opt = torch.div(4,(torch.sqrt(Lh)+torch.sqrt(lh)+epsilon)**2)
                beta_opt = torch.div((torch.sqrt(Lh)-torch.sqrt(lh)),(torch.sqrt(Lh)+torch.sqrt(lh)+epsilon))**2
                
                delta = -alpha_opt*d_p + beta_opt*dx
                
                prev_p.mul_(0).add_(p)
                
                p.add_(delta)

        return loss