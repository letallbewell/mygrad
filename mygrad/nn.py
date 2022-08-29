import random
random.seed(0)

from .engine import Value

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            
    def parameters(self):
        return []

class Neuron(Module):
    
    def __init__(self, n_inputs, non_linearity=True):
        self.W = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0)
        self.non_linearity = non_linearity
        
    def __call__(self, x):
#         assert len(self.W) == len(x)
        activation = sum([Wi * xi for Wi, xi in zip(self.W, x)], self.b)
        activation = activation.ReLU() if self.non_linearity else activation
        
        
        return activation
    
    def parameters(self):
        return self.W + [self.b]
    
    def __repr__(self):
        return f"{'tanh' if self.non_linearity else 'Linear'} Neuron({len(self.W)})"
        
class Layer(Module):
    
    def __init__(self, n_inputs, n_outputs, non_linearity=False):
        self.non_linearity = non_linearity
        self.neurons = [Neuron(n_inputs, non_linearity=self.non_linearity) for _ in range(n_outputs)]
        
    def __call__(self, x):
        output = [neuron(x) for neuron in self.neurons]
        return output[0] if len(output) == 1 else output
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    
    def __init__(self, n_inputs, n_outputs_list):
        sizes = [n_inputs] + n_outputs_list
        self.layers = [Layer(sizes[i], sizes[i+1], non_linearity=True)  #i!=(len(n_outputs_list)-1))
                       for i in range(len(n_outputs_list))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"