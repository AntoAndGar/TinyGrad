from ast import Param
import math
from turtle import back


class Parameter:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Parameter: {self.data}| Grad: {self.grad} | Children: {self._prev}"

    def __add__(self, other):
        other = other if isinstance(other, Parameter) else Parameter(other)
        out = Parameter(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Parameter) else Parameter(other)
        out = Parameter(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, x):
        out = Parameter(self.data**x, (self,), f"**{x}")

        def _backward():
            self.grad += x * (self.data ** (x - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * self**-1

    def tanh(self):
        out = Parameter(
            ((math.e ** (2 * self.data) - 1) / (1 + math.e ** (2 * self.data))),
            (self,),
            "tanh",
        )

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Parameter(0 if self.data <= 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

###############
# TODO: test if derivative are correct

    def sigmoid(self):
        out = Parameter(1 / (1 + math.e ** (-self.data)), (self,), "sigmoid")

        def _backward():
            self.grad += (1 - out.data) * out.data * out.grad

        out._backward = _backward

        return out

    def c_softplus(self):
        e_x = math.e ** self.data
        out = Parameter(math.log(e_x + 1,math.e)-math.log(2,math.e),(self,), "c_softplus")

        def _backward():
            self.grad += e_x / (e_x + 1) * out.grad # TODO: add derivative of ln
        
        out._backward= _backward

        return out

    def lrelu(self, alpha = 0.01):
        out = Parameter(alpha*self.data if self.data < 0 else self.data, (self,), "LReLU")

        def _backward():
            self.grad += (1 if out.data > 0 else alpha) * out.grad

        out._backward = _backward

        return out

    def elu(self, alpha= 0.01):
        e_x = math.e ** self.data
        out = Parameter(alpha*(e_x -1) if self.data < 0 else self.data, (self,), "ELU")

        def _backward():
            self.grad += (1 if out.data > 0 else alpha*e_x) * out.grad

        out._backward = _backward

        return out

##############################

    def backward(self):
        dag = []
        visited = set()

        def build_dag(n):
            if n not in visited:
                visited.add(n)
                [build_dag(children) for children in n._prev]
                dag.append(n)

        build_dag(self)
        
        self.grad = 1.0
        [n._backward() for n in reversed(dag)]
