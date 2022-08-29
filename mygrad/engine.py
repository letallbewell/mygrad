import math

class Value:
    ''' stores a scalar value and its gradient '''
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        # internal variables for graph construction
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
        output._backward = _backward
        
        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward
        
        return output
    

    
    def __pow__(self, other):
        assert isinstance(other,(int, float))
        output = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data) ** (other-1) * output.grad
        output._backward = _backward
        
        return output    
    
    def ReLU(self):

        output = Value(self.data if self.data > 0 else 0, (self, ), 'ReLU')
        
        def _backward():
            self.grad += output.grad if self.data > 0 else 0
        output._backward = _backward
        
        return output    
    
    def tanh(self):
        
        x = self.data
        e_2x = math.exp(2*x)
        tanh_x = (e_2x-1)/(e_2x+1)
        
        output = Value(tanh_x, (self, ), 'tanh')
        
        def _backward():
            self.grad += 1 - tanh_x**2
        output._backward = _backward
        
        return output    
    
    def backward(self):
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-self)
            
    def __rmul__(self, other):
        return other * self
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'