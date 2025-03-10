import torch
import torch.nn as nn
import torch.nn.functional as F


ALL_OPERATIONS = [
    "zero",           # 0: always 0
    "and",            # 1: a AND b
    "not_implies",    # 2: a AND NOT b
    "a",              # 3: a
    "not_implied_by", # 4: NOT a AND b
    "b",              # 5: b
    "xor",            # 6: a XOR b
    "or",             # 7: a OR b
    "not_or",         # 8: NOT (a OR b)
    "not_xor",        # 9: NOT (a XOR b)
    "not_b",          # 10: NOT b
    "implied_by",     # 11: a OR NOT b
    "not_a",          # 12: NOT a
    "implies",        # 13: NOT a OR b
    "not_and",        # 14: NOT (a AND b)
    "one",            # 15: always 1
]


def bin_op(a, b, op_idx):
    """Apply a specific binary logic operation.
    
    Args:
        a: First input tensor
        b: Second input tensor
        op_idx: Index of the operation to apply (0-15)
    
    Returns:
        Result of applying the operation
    """
    if op_idx == 0:    # zero
        return torch.zeros_like(a)
    elif op_idx == 1:  # and
        return a * b
    elif op_idx == 2:  # not_implies (a AND NOT b)
        return a * (1 - b)
    elif op_idx == 3:  # a
        return a
    elif op_idx == 4:  # not_implied_by (NOT a AND b)
        return (1 - a) * b
    elif op_idx == 5:  # b
        return b
    elif op_idx == 6:  # xor
        return a + b - 2 * a * b
    elif op_idx == 7:  # or
        return a + b - a * b
    elif op_idx == 8:  # not_or
        return 1 - (a + b - a * b)
    elif op_idx == 9:  # not_xor
        return 1 - (a + b - 2 * a * b)
    elif op_idx == 10: # not_b
        return 1 - b
    elif op_idx == 11: # implied_by (a OR NOT b)
        return 1 - b + a * b
    elif op_idx == 12: # not_a
        return 1 - a
    elif op_idx == 13: # implies (NOT a OR b)
        return 1 - a + a * b
    elif op_idx == 14: # not_and
        return 1 - a * b
    elif op_idx == 15: # one
        return torch.ones_like(a)
    else:
        raise ValueError(f"Invalid operation index: {op_idx}")


def bin_op_s(a, b, probs):
    """Apply all binary logic operations with their probabilities.
    
    Args:
        a: First input tensor
        b: Second input tensor
        probs: Probabilities for each operation
        
    Returns:
        Weighted combination of all operations
    """
    result = torch.zeros_like(a)
    
    # weighted sum of gates
    for i in range(16):
        result += probs[..., i] * bin_op(a, b, i)
    
    return result


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
        # connections between two inputs and neurons
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
