import torch
import torch.nn as nn
import torch.nn.functional as F
from .operators import bin_op, bin_op_s


class LogicLayer(nn.Module):
    """
    Differentiable logic gate layer that learns to choose a binary operation
    for each neuron.
    """
    def __init__(self, in_dim, out_dim, device='cpu', connections='random', grad_factor=1.0):
        """
        Initialize the logic layer.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            device: Device to use ('cpu' or 'cuda')
            connections: Connection pattern ('random' or 'unique')
            grad_factor: Factor to scale gradients (helps with vanishing gradients)
        """
        super(LogicLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        
        self.weights = nn.Parameter(torch.randn(out_dim, 16, device=device))
        self.connections = connections
        self.indices = self._get_connections(connections, device)
    
    def _get_connections(self, connections, device):
        indices = torch.randperm(2 * self.out_dim) % self.in_dim
        indices = torch.randperm(self.in_dim)[indices]
        indices = indices.reshape(2, self.out_dim)
        a, b = indices[0], indices[1]
        return a.to(device), b.to(device)
    
    def forward(self, x):
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor [batch_size, in_dim]
            
        Returns:
            Output tensor [batch_size, out_dim]
        """
        assert x.shape[1] == self.in_dim, f"Input dimension mismatch: {x.shape[1]} vs {self.in_dim}"
        
        # a and b are the inputs to a given gate/neuron
        a, b = x[:, self.indices[0]], x[:, self.indices[1]]
        
        if self.training:
            if self.grad_factor != 1.0:
                x = GradFactor.apply(x, self.grad_factor)
            
            # normalize distribution over gates to 1
            gate_probs = F.softmax(self.weights, dim=-1)
            result = torch.zeros(x.shape[0], self.out_dim, device=x.device)
            
            for i in range(self.out_dim):
                result[:, i] = bin_op_s(a[:, i], b[:, i], gate_probs[i])
        else:
            # use the most likely gate at inference
            gate_indices = self.weights.argmax(dim=-1)
            result = torch.zeros(x.shape[0], self.out_dim, device=x.device)
            
            for i in range(self.out_dim):
                result[:, i] = bin_op(a[:, i], b[:, i], gate_indices[i].item())
        
        return result
    
    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, mode={'train' if self.training else 'eval'}"


class GradFactor(torch.autograd.Function):
    """Custom autograd function to scale gradients."""
    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.factor, None
