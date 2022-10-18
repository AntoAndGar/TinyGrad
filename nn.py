import random
from core import Parameter


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, number_of_inputs, act=None):
        self.w = [Parameter(random.uniform(-1, 1)) for _ in range(number_of_inputs)]
        self.b = Parameter(0)
        self.act = act

    def __call__(self, x):
        out = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        return (
            out.relu()
            if self.act == "relu"
            else out.tanh()
            if self.act == "tanh"
            else out
        )

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        act = (
            "ReLU" if self.act == "relu" else "Tanh" if self.act == "tanh" else "Linear"
        )
        return f"{act}_Neuron({len(self.w)})"
