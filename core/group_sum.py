import torch
import torch.nn as nn


class GroupSum(nn.Module):
    """
    GroupSum module to aggregate multiple neurons per output bit.
    """
    def __init__(self, k, tau=1.0, device='cpu'):
        """
        Initialize GroupSum.
        
        Args:
            k: Number of intended real-valued outputs (e.g., number of output bits)
            tau: Temperature parameter for scaling
            device: Device to use
        """
        super(GroupSum, self).__init__()
        self.k = k
        self.tau = tau
        self.device = device
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, n] where n is a multiple of k
            
        Returns:
            Output tensor [batch_size, k]
        """
        assert x.shape[1] % self.k == 0, f"Input dimension {x.shape[1]} must be divisible by k={self.k}"
        # Reshape to [batch_size, k, n/k]
        x_reshaped = x.reshape(x.shape[0], self.k, -1)
        # Sum along the last dimension and scale by tau
        return x_reshaped.sum(dim=2) / self.tau
    
    def extra_repr(self):
        """Additional information in string representation."""
        return f"k={self.k}, tau={self.tau}"
