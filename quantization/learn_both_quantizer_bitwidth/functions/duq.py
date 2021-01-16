from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np 
from collections import OrderedDict


class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_lvs):    
        return input.mul(n_lvs-1).round_().div_(n_lvs-1)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Q_ReLU(nn.Module):
    def __init__(self, act_func=True, inplace=False):
        super(Q_ReLU, self).__init__()
        self.n_lvs = 0
        self.act_func = act_func
        self.inplace = inplace
        self.a = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))
        self.theta_x = Parameter(torch.Tensor(0))

    def initialize(self, n_lvs, offset, diff, ):
        self.n_lvs = torch.Tensor(n_lvs).view(-1,1,1,1,1) if len(n_lvs)>1 \
                     else n_lvs[0]
        self.theta_x = Parameter(torch.ones(len(n_lvs))/len(n_lvs))
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
    
    def forward(self, x, accum_cost):
        N, C, H, W = x.shape
        if self.act_func:
            x = F.relu(x, self.inplace)
        
        if self.n_lvs == 0:
            cost = 32 * H * W
        else:
            a = F.softplus(self.a)
            c = F.softplus(self.c)
            x = F.hardtanh(x / a, 0, 1)
            
            if not isinstance(self.n_lvs, torch.Tensor):
                x = RoundQuant.apply(x, self.n_lvs) * c
                cost = self.n_lvs * H * W
                return x, cost
            else:
                # TODO 1: weighted sum of discretized x  (V)
                # TODO 2: compare speed of for loop <-> batched one
                x = x.repeat(self.n_lvs.shape[0], 1, 1, 1, 1)
                x = RoundQuant.apply(x, self.n_lvs) * c
                softmask_x = F.gumbel_softmax(self.theta_x, tau=1, hard=False)
                softmask_x = softmask_x.view(-1,1,1,1,1)
                x = softmask_x * x
                x = x.sum(dim=0)
                
                cost = (softmask * self.n_lvs).sum() * H * W
                
        return x, cost

        
class Q_ReLU6(Q_ReLU):
    def __init__(self, act_func=True, inplace=False):
        super(Q_ReLU6, self).__init__(act_func, inplace)

    def initialize(self, n_lvs, offset, diff):
        self.n_lvs = torch.Tensor(n_lvs).view(-1,1,1,1,1) if len(n_lvs)>1 \
                     else n_lvs[0]
        self.theta_x = Parameter(torch.ones(len(n_lvs))/len(n_lvs))
        if offset + diff > 6:
            self.a.data.fill_(np.log(np.exp(6)-1))
            self.c.data.fill_(np.log(np.exp(6)-1))
        else:
            self.a.data.fill_(np.log(np.exp(offset + diff)-1))
            self.c.data.fill_(np.log(np.exp(offset + diff)-1))


class Q_Sym(nn.Module):
    def __init__(self):
        super(Q_Sym, self).__init__()
        self.n_lvs = 0
        self.a = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))
        self.theta_x = Parameter(torch.Tensor(0))

    def initialize(self, n_lvs, offset, diff):
        self.n_lvs = torch.Tensor(n_lvs).view(-1,1,1,1,1) if len(n_lvs)>1 \
                     else n_lvs[0]
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
        self.theta_x = Parameter(torch.ones(len(n_lvs))/len(n_lvs))

    def forward(self, x):
        N, C, H, W = x.shape
        if self.n_lvs == 0:
            cost = 32 * H * W
            return x, cost
        else:
            a = F.softplus(self.a)
            c = F.softplus(self.c)
            x = F.hardtanh(x / a, -1, 1)

            if not instance(self.n_lvs, torch.Tensor):
                x = RoundQuant.apply(x, self.n_lvs // 2) * c
                cost = self.n_lvs * H * W
                return x, cost
            else:
                x = x.repeat(self.n_lv.shape[0], 1, 1, 1, 1)
                x = RoundQuant.apply(x, self.n_lvs) * c
                softmask = F.gumbel_softmax(self.theta_x, tau=1, hard=False)
                softmask = softmask.view(-1,1,1,1,1)
                x = softmask_x * x
                x = x.sum(dim=0)
                cost = (softmask * self.n_lvs).sum() * H * W
                return x, cost 


class Q_HSwish(nn.Module):
    def __init__(self, act_func=True):
        super(Q_HSwish, self).__init__()
        self.n_lvs = 0
        self.act_func = act_func
        self.a = Parameter(torch.Tensor(1))
        self.b = 3/8
        self.c = Parameter(torch.Tensor(1))
        self.d = -3/8

    def initialize(self, n_lv, offset, diff):
        self.n_lvs = n_lvs
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
    
    def forward(self, x):
        if self.act_func:
            x = x * (F.hardtanh(x + 3, 0, 6) / 6)

        if self.n_lvs == 0:
            return x
        else:
            a = F.softplus(self.a)
            c = F.softplus(self.c)
            x = x + self.b
            x = F.hardtanh(x / a, 0, 1)
            x = RoundQuant.apply(x, self.n_lvs) * c
            x = x + self.d
            return x 


class Q_Conv2d(nn.Conv2d):
    def __init__(self, *args, **kargs):
        super(Q_Conv2d, self).__init__(*args, **kargs)
        self.n_lvs = 0
        self.a = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))
        self.weight_old = None
        self.theta_w = Parameter(torch.Tensor(0))

    def initialize(self, n_lvs):
        self.n_lvs = torch.Tensor(n_lvs).view(-1,1,1,1,1) if len(n_lvs)>1 \
                     else n_lvs[0]
        self.theta_w = Parameter(torch.ones(len(n_lvs))/len(n_lvs))
        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))

    def _weight_quant(self):
        a = F.softplus(self.a)
        c = F.softplus(self.c)
        weight = F.hardtanh(self.weight / a, -1, 1)

        if not isinstance(self.n_lvs, torch.Tensor):
            weight = RoundQuant.apply(weight, self.n_lvs // 2) * c
            bitwidth = self.n_lvs
        else:
            weight = weight.repeat(self.n_lvs.shape[0], 1, 1, 1, 1)
            weight = RoundQuant.apply(weight, self.n_lvs // 2) * c
            softmask = F.gumbel_softmax(self.theta_w, tau=1, hard=False)
            softmask = softmask.view(-1,1,1,1,1)
            weight = softmask * weight
            weight = weight.sum(dim=0)
            bitwidth = (softmask * self.n_lvs).sum()

        return weight, bitwidth

    def forward(self, x, act_cost):
        O, I, K1, K2 = self.weight.shape
        if self.n_lvs == 0:
            cost = act_cost * 32 * O * I * K1 * K2 * (1/self.stride) * (1/self.stride)
            return F.conv2d(x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups), cost
        else:
            weight, bitwidth = self._weight_quant()
            cost = act_cost * bitwidth * O * I * K1 * K2 * (1/self.stride) * (1/self.stride)
            return F.conv2d(x, weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups), cost


class Q_Linear(nn.Linear):
    def __init__(self, *args, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.n_lvs = 0
        self.a = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))
        self.weight_old = None

    def initialize(self, n_lvs):
        self.n_lvs = torch.Tensor(n_lvs).view(-1,1,1,1,1) if len(n_lvs)>1 \
                     else n_lvs[0]
        self.theta_w = Parameter(torch.ones(len(n_lvs))/len(n_lvs))
        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))

    def _weight_quant(self, act_cost):
        a = F.softplus(self.a)
        c = F.softplus(self.c)

        weight = F.hardtanh(self.weight / a, -1, 1)
        if not isinstance(self.n_lvs, torch.Tensor):
            weight = RoundQuant.apply(weight, self.n_lvs // 2) * c
            bitwidth = self.n_lvs
        else:
            weight = weight.repeat(self.n_lv.shape[0], 1, 1, 1, 1)
            weight = RoundQuant.apply(weight, self.n_lvs // 2) * c
            softmask = F.gumbel_softmax(self.theta_w, tau=1, hard=False)
            softmask = softmask.view(-1,1,1,1,1)
            weight = softmask * weight
            weight = weight.sum(dim=0)
            bitwidth = (softmask * self.n_lvs).sum()
        return weight, cost

    def forward(self, x):
        O, I = self.weight.shape
        if self.n_lvs == 0:
            cost = act_cost * 32 * O * I
            return F.linear(x, self.weight, self.bias), cost
        else:
            weight = self._weight_quant()
            cost = act_cost * bitwidth * O * I
            return F.linear(x, weight, self.bias), cost


class Q_Conv2dPad(Q_Conv2d):
    def __init__(self, mode, *args, **kargs):
        super(Q_Conv2dPad, self).__init__(*args, **kargs)
        self.mode = mode

    def forward(self, inputs):
        if self.mode == "HS":
            inputs = F.pad(inputs, self.padding + self.padding, value=-3/8)
        elif self.mode == "RE":
            inputs = F.pad(inputs, self.padding + self.padding, value=0)
        else:
            raise LookupError("Unknown nonlinear")

        if self.n_lvs == 0:
            return F.conv2d(inputs, self.weight, self.bias,
                self.stride, 0, self.dilation, self.groups)
        else:
            weight = self._weight_quant()

            return F.conv2d(inputs, weight, self.bias,
                self.stride, 0, self.dilation, self.groups)



def initialize(model, loader, n_lvs, act=False, weight=False, eps=0.05):
    def initialize_hook(module, input, output):
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)) and act:
            if not isinstance(input, torch.Tensor):
                input = input[0]
            input = input.detach().cpu().numpy()

            if isinstance(input, Q_Sym):
                input = np.abs(input)
            elif isinstance(input, Q_HSwish):
                input = input + 3/8

            input = input.reshape(-1)
            input = input[input > 0]
            input = np.sort(input)
            
            if len(input) == 0:
                small, large = 0, 1e-3
            else:
                small, large = input[int(len(input) * eps)], input[int(len(input) * (1-eps))]

            module.initialize(n_lvs, small, large - small)

        if isinstance(module, (Q_Conv2d, Q_Linear)) and weight:
            module.initialize(n_lvs)

    hooks = []

    for name, module in model.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)


    model.train()
    model.cpu()
    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                output = model.module(input)
            else:
                output = model(input)
        break
    
    model.cuda()
    for hook in hooks:
        hook.remove()


class Q_Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Q_Sequential, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            idx = 0 
            for module in args:
                if isinstance(module, Q_Sym) or (isinstance(module, Q_HSwish) and idx == 0):
                    self.add_module('-' + str(idx), module)
                else:
                    self.add_module(str(idx), module)
                    idx += 1


class QuantOps(object):
    initialize = initialize
    Conv2d = Q_Conv2d
    ReLU = Q_ReLU
    ReLU6 = Q_ReLU6
    Sym = Q_Sym
    HSwish = Q_HSwish
    Conv2dPad = Q_Conv2dPad        
    Sequential = Q_Sequential
    Linear = Q_Linear