import os
import torch
import argparse
from datetime import datetime

from utils.config import load_config, save_config
from utils.training import train_model, evaluate_model
from core.model import LogicGateNetwork
from problems import get_problem


def main(args):
    config = load_config(args.config)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['problem']['name']}_{timestamp}"
    experiment_dir = os.path.join(config['output_dir'], experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    save_config(config, os.path.join(experiment_dir, 'config.yaml'))
    
    problem = get_problem(config['problem']['name'], config['problem'])
    
    train_loader, val_loader = problem.create_data_loaders(
        train_ratio=config['training']['train_ratio'],
        batch_size=config['training']['batch_size']
    )
    
    model = LogicGateNetwork(
        input_dim=problem.input_dim,
        hidden_dims=config['model']['hidden_dims'],
        output_dim=problem.output_dim,
        device=config['model']['device'],
        grad_factor=config['model']['grad_factor']
    )
    
    print(f"\nModel summary:")
    print(f"  Input dimension: {problem.input_dim}")
    print(f"  Output dimension: {problem.output_dim}")
    print(f"  Hidden dimensions: {config['model']['hidden_dims']}")
    print(f"  Total neurons: {sum(config['model']['hidden_dims']) + problem.output_dim}")
    print("")
    
    if not args.skip_training:
        print("Starting training...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config['training'],
            save_dir=os.path.join(experiment_dir, 'checkpoints')
        )
        
    else:
        checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best_model.pt')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded pre-trained model from {checkpoint_path}")
        else:
            print("No pre-trained model found. Skipping training.")
    
    print("\nEvaluating model...")
    results = evaluate_model(model, val_loader)
    
    print("\nEvaluation results:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Example accuracy: {results['example_accuracy']:.4f}")
    print(f"  Bit accuracy: {results['bit_accuracy']:.4f}")
    print("  Per-bit accuracy:")
    for i, acc in enumerate(results['per_bit_accuracy']):
        print(f"    Bit {i}: {acc:.4f}")
    
    gate_analysis = model.analyze_gates()
    print("\nGate usage analysis:")
    for layer_name, gates in gate_analysis.items():
        if layer_name == 'overall':
            continue
        print(f"  {layer_name}:")
        for gate_name, count in gates.items():
            print(f"    {gate_name}: {count}")
    
    print("\nOverall gate usage:")
    for gate_name, count in gate_analysis['overall'].items():
        print(f"  {gate_name}: {count}")
    
    print(f"\nExperiment results saved to: {experiment_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logic Gate Networks for Program Synthesis")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and load pre-trained model if available')
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip visualization of gate usage')
    
    args = parser.parse_args()
    main(args)
