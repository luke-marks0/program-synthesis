import torch
import torch.nn as nn
from .logic_layer import LogicLayer
from .operators import ALL_OPERATIONS
from .group_sum import GroupSum


class LogicGateNetwork(nn.Module):
    """
    Complete network of logic gate layers for program synthesis tasks.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 neurons_per_output=16, tau=1.0,
                 connections='random', device='cpu', grad_factor=1.0):
        """
        Initialize a logic gate network.
        
        Args:
            input_dim: Dimension of input
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (number of output bits)
            neurons_per_output: Number of neurons per output bit
            tau: Temperature parameter for scaling
            connections: Connection pattern ('random' or 'unique')
            device: Device to use ('cpu' or 'cuda')
            grad_factor: Gradient scaling factor
        """
        super(LogicGateNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.neurons_per_output = neurons_per_output
        self.tau = tau
        self.device = device
        
        layers = []
        prev_dim = input_dim
        total_output_neurons = output_dim * neurons_per_output
        
        for i, dim in enumerate(hidden_dims):
            # scale grad factor with layer depth
            layer_grad_factor = grad_factor * (1.5 ** i) if i > 0 else grad_factor
            layer = LogicLayer(prev_dim, dim, device, connections, layer_grad_factor)
            layers.append(layer)
            prev_dim = dim
        
        # output layer
        layers.append(LogicLayer(prev_dim, output_dim, device, connections, 
                                grad_factor * (1.5 ** len(hidden_dims))))
        
        self.layers = nn.ModuleList(layers)

        self.group_sum = GroupSum(output_dim, tau, device)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        for layer in self.layers:
            x = layer(x)

        x = self.group_sum(x)

        return x
    
    def get_gates(self):
        """
        Get the most likely gate for each neuron in each layer.
        
        Returns:
            List of tensors with gate indices for each layer
        """
        gates = []
        
        for layer in self.layers:
            gate_indices = layer.weights.argmax(dim=-1)
            gates.append(gate_indices)
        
        return gates
    
    def analyze_gates(self):
        """
        Analyze the gate usage in the network.
        
        Returns:
            Dictionary with gate usage statistics for each layer
        """
        gates = self.get_gates()
        results = {}
        
        for i, layer_gates in enumerate(gates):
            layer_name = f"layer_{i+1}"
            gate_counts = {}
            
            for op_idx, op_name in enumerate(ALL_OPERATIONS):
                count = (layer_gates == op_idx).sum().item()
                if count > 0:
                    gate_counts[op_name] = count
            
            results[layer_name] = gate_counts
        
        all_gates = torch.cat([g.flatten() for g in gates])
        overall_counts = {}
        
        for op_idx, op_name in enumerate(ALL_OPERATIONS):
            count = (all_gates == op_idx).sum().item()
            if count > 0:
                overall_counts[op_name] = count
        
        results['overall'] = overall_counts
        
        return results
