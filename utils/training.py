import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_model(model, train_loader, val_loader, config, save_dir=None):
    """
    Train a logic gate network.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        save_dir: Directory to save checkpoints
        
    Returns:
        Dictionary with training history
    """
    num_epochs = config.get('num_epochs', 100)
    learning_rate = config.get('learning_rate', 0.01)
    weight_decay = config.get('weight_decay', 0.0)
    patience = config.get('patience', 20)
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=patience // 3,
    )
    
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_bit_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_bit_acc': [],
        'epoch_times': []
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Starting training for {num_epochs} epochs on {device}...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        model.train()
        train_loss = 0.0
        correct_examples = 0
        correct_bits = 0
        total_examples = 0
        total_bits = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_examples += (predictions == targets).all(dim=1).sum().item()
            correct_bits += (predictions == targets).sum().item()
            total_examples += inputs.size(0)
            total_bits += targets.numel()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct_examples / total_examples
            })
        
        train_loss = train_loss / total_examples
        train_acc = correct_examples / total_examples
        train_bit_acc = correct_bits / total_bits
        
        model.eval()
        val_loss = 0.0
        correct_examples = 0
        correct_bits = 0
        total_examples = 0
        total_bits = 0
        
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        
        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct_examples += (predictions == targets).all(dim=1).sum().item()
                correct_bits += (predictions == targets).sum().item()
                
                total_examples += inputs.size(0)
                total_bits += targets.numel()
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': correct_examples / total_examples
                })
        
        val_loss = val_loss / total_examples
        val_acc = correct_examples / total_examples
        val_bit_acc = correct_bits / total_bits
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1}")
            break
        
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_bit_acc'].append(train_bit_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_bit_acc'].append(val_bit_acc)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Train Bit Acc: {train_bit_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val Bit Acc: {val_bit_acc:.4f}")
    
    if save_dir and os.path.exists(os.path.join(save_dir, 'best_model.pt')):
        model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    
    return history


def evaluate_model(model, test_loader):
    """
    Evaluate a trained model.
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    correct_examples = 0
    correct_bits = 0
    total_examples = 0
    total_bits = 0
    
    criterion = nn.BCEWithLogitsLoss()
    
    progress_bar = tqdm(test_loader, desc="Evaluation")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            correct_examples += (predictions == targets).all(dim=1).sum().item()
            correct_bits += (predictions == targets).sum().item()
            
            total_examples += inputs.size(0)
            total_bits += targets.numel()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct_examples / total_examples
            })
    
    test_loss = test_loss / total_examples
    test_acc = correct_examples / total_examples
    test_bit_acc = correct_bits / total_bits
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    per_bit_acc = []
    for i in range(all_targets.shape[1]):
        bit_acc = (all_preds[:, i] == all_targets[:, i]).float().mean().item()
        per_bit_acc.append(bit_acc)
    
    results = {
        'loss': test_loss,
        'example_accuracy': test_acc,
        'bit_accuracy': test_bit_acc,
        'per_bit_accuracy': per_bit_acc
    }
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Example Accuracy: {test_acc:.4f}")
    print(f"Test Bit Accuracy: {test_bit_acc:.4f}")
    
    return results
