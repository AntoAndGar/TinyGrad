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

class Layer(Module):
    def __init__(self, nin, non, act=None):
        self.layer = [Neuron(nin, act) for _ in range(non)]

    def __call__(self, input):
        out = [n(input) for n in self.layer]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.layer for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.layer)}]"

class MLP(Module):
    def __init__(self, nin, nouts, act):
        sizes = [nin] + nouts
        []
        self.layers = [
            Layer(sizes[i], sizes[i + 1], act=act if isinstance(act, str) else act[i] if i<len(act) else None)
            for i in range(len(nouts))
        ]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
