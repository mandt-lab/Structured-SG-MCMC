"""Taken and partially adapted from: 
- https://github.com/henripal/sgld/blob/master/sgld/sgld/sgld_optimizer.py
- https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Stochastic_Gradient_HMC_SA/optimizers.py"""

import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
import numpy as np


class SGLD(Optimizer):
    """
    Barely modified version of pytorch SGD to implement SGLD
    """

    def __init__(self, params, lr=required, addnoise=True):
        defaults = dict(lr=lr, addnoise=addnoise)
        super(SGLD, self).__init__(params, defaults)

    def step(self, lr=None, add_noise=False):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['addnoise']:
                    #size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros_like(d_p),
                        torch.ones_like(d_p) / np.sqrt(group['lr'])
                    )
                    p.data.add_(
                        d_p + langevin_noise.sample(), # .cuda(),
                        alpha=-group['lr'],
                    )
                else:
                    p.data.add_(d_p, alpha=-group['lr'])

        return loss
    
class pSGLD(Optimizer):
    """
    Barely modified version of pytorch SGD to implement pSGLD
    The RMSprop preconditioning code is mostly from pytorch rmsprop implementation.
    """

    def __init__(self, params, lr=required, alpha=0.99, eps=1e-8, centered=False, addnoise=True):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, centered=centered, addnoise=addnoise)
        super(pSGLD, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(pSGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self, lr=None, add_noise=False):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)
                        
                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1
                
                # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                square_avg.mul_(alpha).addcmul_(1-alpha, d_p, d_p)
                
                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1-alpha, d_p)
                    avg = square_avg.cmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])
                    
                
                if group['addnoise']:
                    #size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros_like(d_p),
                        torch.ones_like(d_p).div_(group['lr']).div_(avg).sqrt(),
                    )
                    p.data.add_(
                        d_p.div_(avg) + langevin_noise.sample(),
                        alpha=-group['lr'],
                    )
                else:
                    #p.data.add_(-group['lr'], d_p.div_(avg))
                    #p.data.addcdiv_(d_p, avg, value=-group['lr'])
                    p.data.add_(
                        d_p.div_(avg),
                        alpha=-group['lr'],
                    )

        return loss

class H_SA_SGHMC(Optimizer):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses scale adaption during burn-in
        procedure to find some hyperparamters."""

    def __init__(self, params, lr=1e-2, base_C=0.05, burn_in_period=150, momentum_sample_freq=1000, addnoise=True):
        self.eps = 1e-6
        self.burn_in_period = burn_in_period
        self.momentum_sample_freq = momentum_sample_freq
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if base_C < 0:
            raise ValueError("Invalid friction term: {}".format(base_C))

        defaults = dict(
            lr=lr,
            base_C=base_C,
            addnoise=addnoise,
        )
        super(H_SA_SGHMC, self).__init__(params, defaults)

    def step(self):
        """Simulate discretized Hamiltonian dynamics for one step"""
        loss = None

        for group in self.param_groups:  # iterate over blocks -> the ones defined in defaults. We dont use groups.
            for p in group["params"]:  # these are weight and bias matrices
                if p.grad is None:
                    continue
                state = self.state[p]  # define dict for each individual param
                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(p)
                    state["g"] = torch.ones_like(p)
                    state["V_hat"] = torch.ones_like(p)
                    state["v_momentum"] = torch.zeros_like(
                        p)  # p.data.new(p.data.size()).normal_(mean=0, std=np.sqrt(group["lr"])) #

                state["iteration"] += 1  # this is kind of useless now but lets keep it provisionally

                base_C, lr = group["base_C"], group["lr"]
                tau, g, V_hat = state["tau"], state["g"], state["V_hat"]

                d_p = p.grad.data

                # update parameters during burn-in
                if state["iteration"] <= self.burn_in_period: # We update g first as it makes most sense
                    tau.add_(-tau * (g ** 2) / (
                                V_hat + self.eps) + 1)  # specifies the moving average window, see Eq 9 in [1] left
                    tau_inv = 1. / (tau + self.eps)
                    g.add_(-tau_inv * g + tau_inv * d_p)  # average gradient see Eq 9 in [1] right
                    V_hat.add_(-tau_inv * V_hat + tau_inv * (d_p ** 2))  # gradient variance see Eq 8 in [1]

                V_sqrt = torch.sqrt(V_hat)
                V_inv_sqrt = 1. / (V_sqrt + self.eps)  # preconditioner

                if (state["iteration"] % self.momentum_sample_freq) == 0:  # equivalent to var = M under momentum reparametrisation
                    state["v_momentum"] = torch.normal(mean=torch.zeros_like(d_p),
                                                       std=torch.sqrt((lr ** 2) * V_inv_sqrt))
                v_momentum = state["v_momentum"]

                if group['addnoise']:
                    noise_var = (2. * (lr ** 2) * V_inv_sqrt * base_C - (lr ** 4))
                    noise_std = torch.sqrt(torch.clamp(noise_var, min=1e-16))
                    # sample random epsilon
                    noise_sample = torch.normal(mean=torch.zeros_like(d_p), std=torch.ones_like(d_p) * noise_std)
                    # update momentum (Eq 10 right in [1])
                    v_momentum.add_(- (lr ** 2) * V_inv_sqrt * d_p - base_C * v_momentum + noise_sample)
                else:
                    # update momentum (Eq 10 right in [1])
                    v_momentum.add_(- (lr ** 2) * V_inv_sqrt * d_p - base_C * v_momentum)

                # update theta (Eq 10 left in [1])
                p.data.add_(v_momentum)

        return loss
