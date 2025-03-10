import os
import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader


class ProgramSynthesisDataset(Dataset):
    """Dataset for program synthesis tasks with binary vectors."""
    def __init__(self, inputs, outputs):
        """
        Initialize the dataset.
        
        Args:
            inputs: Binary input vectors [n_samples, input_dim]
            outputs: Binary output vectors [n_samples, output_dim]
        """
        self.inputs = inputs
        self.outputs = outputs
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class ProgramSynthesisProblem(ABC):
    """Base class for program synthesis problems."""
    def __init__(self, config):
        """
        Initialize the problem.
        
        Args:
            config: Problem configuration dictionary
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__.lower())
        self.data_dir = config.get('data_dir', 'data')
        
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_dataset_path(self):
        """Get the path to the dataset file."""
        params_str = f"_{self.config.get('bits', 0)}"
        return os.path.join(self.data_dir, f"{self.name}{params_str}_dataset.pt")
    
    def dataset_exists(self):
        """Check if the dataset exists."""
        return os.path.exists(self.get_dataset_path())
    
    def load_dataset(self):
        """Load the dataset from disk."""
        data = torch.load(self.get_dataset_path())
        
        if data['inputs'].shape[1] != self.input_dim or data['outputs'].shape[1] != self.output_dim:
            print(f"Warning: Dataset dimensions don't match current configuration.")
            print(f"Expected: inputs ({self.input_dim}), outputs ({self.output_dim})")
            print(f"Found: inputs ({data['inputs'].shape[1]}), outputs ({data['outputs'].shape[1]})")
            print(f"Regenerating dataset...")
            return None
            
        return ProgramSynthesisDataset(data['inputs'], data['outputs'])
    
    def save_dataset(self, inputs, outputs):
        """Save the dataset to disk."""
        data = {
            'inputs': inputs,
            'outputs': outputs,
            'config': {
                'bits': self.config.get('bits', 0),
                'num_samples': self.config.get('num_samples', 0)
            }
        }
        torch.save(data, self.get_dataset_path())
    
    def get_or_create_dataset(self):
        """Get the dataset, creating it if it doesn't exist."""
        if self.dataset_exists():
            print(f"Loading existing dataset for {self.name}...")
            dataset = self.load_dataset()
            if dataset is not None:
                return dataset
        
        print(f"Generating dataset for {self.name}...")
        inputs, outputs = self.generate_dataset()
        self.save_dataset(inputs, outputs)
        return ProgramSynthesisDataset(inputs, outputs)
    
    def create_data_loaders(self, train_ratio=0.8, batch_size=32, shuffle=True):
        """
        Create training and validation data loaders.
        
        Args:
            train_ratio: Ratio of data to use for training
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        dataset = self.get_or_create_dataset()
        
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    @abstractmethod
    def generate_dataset(self):
        """
        Generate a dataset for this problem.
        
        Returns:
            Tuple of (inputs, outputs) tensors
        """
        pass
    
    @property
    @abstractmethod
    def input_dim(self):
        """Get the input dimension."""
        pass
    
    @property
    @abstractmethod
    def output_dim(self):
        """Get the output dimension."""
        pass


class BinaryAdder(ProgramSynthesisProblem):
    """
    Binary adder problem.
    
    Input: Two n-bit binary numbers
    Output: (n+1)-bit binary sum
    """
    def __init__(self, config):
        super().__init__(config)
        self.bits = config.get('bits', 4)
    
    @property
    def input_dim(self):
        return 2 * self.bits
    
    @property
    def output_dim(self):
        return self.bits + 1  # sum can have n+1 bits
    
    def generate_dataset(self):
        """Generate dataset of binary addition examples."""
        num_samples = self.config.get('num_samples', 1000)
        total_possible = 2**(2*self.bits)
        
        if num_samples >= total_possible:
            print(f"Generating complete dataset with {total_possible} examples (all possible combinations)")
            a_values = torch.arange(2**self.bits)
            b_values = torch.arange(2**self.bits)
            
            a_grid, b_grid = torch.meshgrid(a_values, b_values, indexing='ij')
            a_flat = a_grid.flatten()
            b_flat = b_grid.flatten()
        else:
            print(f"Generating random dataset with {num_samples} examples (out of {total_possible} possible)")
            a_flat = torch.randint(0, 2**self.bits, (num_samples,))
            b_flat = torch.randint(0, 2**self.bits, (num_samples,))
        
        a_binary = self._to_binary(a_flat, self.bits)
        b_binary = self._to_binary(b_flat, self.bits)
        
        sum_values = a_flat + b_flat
        sum_binary = self._to_binary(sum_values, self.bits + 1)
        
        inputs = torch.cat([a_binary, b_binary], dim=1)
        
        return inputs, sum_binary
    
    def _to_binary(self, values, num_bits):
        """Convert integer values to binary representation."""
        binary = torch.zeros(values.size(0), num_bits)
        
        for i in range(num_bits):
            binary[:, num_bits - 1 - i] = ((values >> i) & 1).float()
        
        return binary


class BinaryMultiplier(ProgramSynthesisProblem):
    """
    Binary multiplier problem.
    
    Input: Two n-bit binary numbers
    Output: (2n)-bit binary product
    """
    def __init__(self, config):
        super().__init__(config)
        self.bits = config.get('bits', 4)
    
    @property
    def input_dim(self):
        return 2 * self.bits
    
    @property
    def output_dim(self):
        return 2 * self.bits  # product can have 2n bits
    
    def generate_dataset(self):
        """Generate dataset of binary multiplication examples."""
        num_samples = self.config.get('num_samples', 1000)
        total_possible = 2**(2*self.bits)
        
        if num_samples >= total_possible:
            print(f"Generating complete dataset with {total_possible} examples (all possible combinations)")
            a_values = torch.arange(2**self.bits)
            b_values = torch.arange(2**self.bits)
            
            a_grid, b_grid = torch.meshgrid(a_values, b_values, indexing='ij')
            a_flat = a_grid.flatten()
            b_flat = b_grid.flatten()
        else:
            print(f"Generating random dataset with {num_samples} examples (out of {total_possible} possible)")
            a_flat = torch.randint(0, 2**self.bits, (num_samples,))
            b_flat = torch.randint(0, 2**self.bits, (num_samples,))
        
        a_binary = self._to_binary(a_flat, self.bits)
        b_binary = self._to_binary(b_flat, self.bits)
        
        product_values = a_flat * b_flat
        product_binary = self._to_binary(product_values, 2 * self.bits)
        
        inputs = torch.cat([a_binary, b_binary], dim=1)
        
        return inputs, product_binary
    
    def _to_binary(self, values, num_bits):
        """Convert integer values to binary representation."""
        binary = torch.zeros(values.size(0), num_bits)
        
        for i in range(num_bits):
            binary[:, num_bits - 1 - i] = ((values >> i) & 1).float()
        
        return binary


class BinaryParity(ProgramSynthesisProblem):
    """
    Binary parity problem.
    
    Input: n-bit binary number
    Output: 1 if number of 1s is odd, 0 if even
    """
    def __init__(self, config):
        super().__init__(config)
        self.bits = config.get('bits', 8)
    
    @property
    def input_dim(self):
        return self.bits
    
    @property
    def output_dim(self):
        return 1
    
    def generate_dataset(self):
        """Generate dataset of binary parity examples."""
        num_samples = self.config.get('num_samples', 1000)
        total_possible = 2**self.bits
        
        if num_samples >= total_possible:
            print(f"Generating complete dataset with {total_possible} examples (all possible combinations)")
            values = torch.arange(2**self.bits)
        else:
            print(f"Generating random dataset with {num_samples} examples (out of {total_possible} possible)")
            values = torch.randint(0, 2**self.bits, (num_samples,))
        
        inputs = self._to_binary(values, self.bits)
        
        num_ones = torch.sum(inputs, dim=1)
        parity = (num_ones % 2).view(-1, 1)
        
        return inputs, parity
    
    def _to_binary(self, values, num_bits):
        """Convert integer values to binary representation."""
        binary = torch.zeros(values.size(0), num_bits)
        
        for i in range(num_bits):
            binary[:, num_bits - 1 - i] = ((values >> i) & 1).float()
        
        return binary


class BinarySorter(ProgramSynthesisProblem):
    """
    Binary sorter problem.
    
    Input: n-bit binary numbers
    Output: Sorted bits (all 0s followed by all 1s)
    """
    def __init__(self, config):
        super().__init__(config)
        self.bits = config.get('bits', 8)
    
    @property
    def input_dim(self):
        return self.bits
    
    @property
    def output_dim(self):
        return self.bits
    
    def generate_dataset(self):
        """Generate dataset of binary sorting examples."""
        num_samples = self.config.get('num_samples', 1000)
        total_possible = 2**self.bits
        
        if num_samples >= total_possible:
            print(f"Generating complete dataset with {total_possible} examples (all possible combinations)")
            values = torch.arange(2**self.bits)
        else:
            print(f"Generating random dataset with {num_samples} examples (out of {total_possible} possible)")
            values = torch.randint(0, 2**self.bits, (num_samples,))
        
        inputs = self._to_binary(values, self.bits)
        
        num_ones = torch.sum(inputs, dim=1, keepdim=True)
        num_zeros = self.bits - num_ones
        
        outputs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outputs[i, int(num_zeros[i]):] = 1.0
        
        return inputs, outputs
    
    def _to_binary(self, values, num_bits):
        """Convert integer values to binary representation."""
        binary = torch.zeros(values.size(0), num_bits)
        
        for i in range(num_bits):
            binary[:, num_bits - 1 - i] = ((values >> i) & 1).float()
        
        return binary


PROBLEMS = {
    'adder': BinaryAdder,
    'multiplier': BinaryMultiplier,
    'parity': BinaryParity,
    'sorter': BinarySorter
}

def get_problem(name, config):
    """
    Get a problem instance by name.
    
    Args:
        name: Problem name
        config: Problem configuration
    
    Returns:
        ProgramSynthesisProblem instance
    """
    if name not in PROBLEMS:
        raise ValueError(f"Unknown problem: {name}. Available problems: {list(PROBLEMS.keys())}")
    
    return PROBLEMS[name](config)
