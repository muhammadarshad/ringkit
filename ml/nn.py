"""
ring_nn.py — RingModule + a simple neuron (SRD Phase 5, T5.1), on ring_autograd/optim.

Parameters are persistent autograd Vars (ARC values, ENERGY grads). Forward builds the
graph via scalar Var ops (a dot product is a sum of muls — matmul unrolled through the
scalar engine, so backward Just Works). step() updates each parameter with ring_optim.
Multiplier-free.
"""
from ringkit.ml import autograd as ag
from ringkit.ml import optim as opt


class RingModule:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def step(self, lr=1):
        for p in self.parameters():
            p.val = opt.sgd_step(p.val, p.grad, lr)


class Neuron(RingModule):
    """out(x) = SIN( sum_i W_i * x_i + b ). Weights/bias are learnable ARC params."""

    def __init__(self, weights, bias):
        self.W = [ag.Var(w) for w in weights]
        self.b = ag.Var(bias)

    def parameters(self):
        return self.W + [self.b]

    def forward(self, x):
        acc = self.b
        for wi, xi in zip(self.W, x):
            acc = acc.add(wi.mul(ag.Var(xi)))
        return acc.sin()
