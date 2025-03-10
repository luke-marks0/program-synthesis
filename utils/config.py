import os
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set default values if not specified
    config = set_default_values(config)
    
    return config


def set_default_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set default values for missing configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Updated configuration dictionary
    """
    # Default directories
    config.setdefault('data_dir', 'data')
    config.setdefault('output_dir', 'output')
    config.setdefault('models_dir', os.path.join(config['output_dir'], 'models'))
    config.setdefault('plots_dir', os.path.join(config['output_dir'], 'plots'))
    
    # Create directories if they don't exist
    for dir_path in [config['data_dir'], config['output_dir'], 
                     config['models_dir'], config['plots_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Default problem settings
    if 'problem' not in config:
        config['problem'] = {}
    
    config['problem'].setdefault('name', 'adder')
    config['problem'].setdefault('bits', 4)
    config['problem'].setdefault('num_samples', 1000)
    
    # Default model settings
    if 'model' not in config:
        config['model'] = {}
    
    config['model'].setdefault('hidden_dims', [32, 32])
    config['model'].setdefault('connections', 'random')
    config['model'].setdefault('device', 'cpu')
    config['model'].setdefault('grad_factor', 1.0)
    
    # Default training settings
    if 'training' not in config:
        config['training'] = {}
    
    config['training'].setdefault('num_epochs', 100)
    config['training'].setdefault('batch_size', 32)
    config['training'].setdefault('learning_rate', 0.01)
    config['training'].setdefault('weight_decay', 0.0)
    config['training'].setdefault('patience', 20)
    config['training'].setdefault('train_ratio', 0.8)
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
