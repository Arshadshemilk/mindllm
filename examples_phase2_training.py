"""
Example script for Phase 2: Training confidence classifiers offline.

This script demonstrates:
1. Generating synthetic training data
2. Training confidence heads with geometric loss
3. Calibrating predictions with temperature scaling
4. Evaluating Expected Calibration Error (ECE)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.phase2_confidence_classifier.confidence_heads import (
    ConfidenceClassifierEnsemble,
    ConfidenceTrainer
)
from src.config import ConfidenceClassifierConfig, ModelConfig
from src.utils.utilities import set_seed


def generate_synthetic_training_data(
    num_samples: int = 1000,
    num_layers: int = 8,
    hidden_dim: int = 2048,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]]]:
    """
    Generate synthetic training data for confidence classifiers.
    
    In a real scenario, this would come from actual forward passes through the model.
    
    Args:
        num_samples: Number of training samples
        num_layers: Number of exit layers
        hidden_dim: Hidden dimension size
        device: Device to use
        
    Returns:
        (hidden_states, targets) tuple
    """
    hidden_states_dict = {}
    targets_dict = {}
    
    exit_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    
    for layer_idx in exit_layers:
        # Generate synthetic hidden states
        # In real scenario: [num_samples, seq_len, hidden_dim]
        hidden_states = [
            torch.randn(1, 32, hidden_dim, device=device)
            for _ in range(num_samples)
        ]
        hidden_states_dict[layer_idx] = hidden_states
        
        # Generate synthetic targets
        # 1 = confident (exit prediction matches final), 0 = uncertain
        targets = [
            torch.bernoulli(torch.ones(1, 32, device=device) * 0.7).unsqueeze(-1)
            for _ in range(num_samples)
        ]
        targets_dict[layer_idx] = targets
    
    return hidden_states_dict, targets_dict


def train_confidence_classifiers():
    """Train confidence classifiers with geometric loss."""
    print("=" * 60)
    print("Training Confidence Classifiers (Phase 2)")
    print("=" * 60)
    
    # Configuration
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model_config = ModelConfig()
    classifier_config = ConfidenceClassifierConfig()
    
    # Initialize model
    print(f"\nInitializing confidence classifier ensemble...")
    ensemble = ConfidenceClassifierEnsemble(
        exit_layer_indices=model_config.exit_layer_indices,
        input_dim=classifier_config.input_dim,
        hidden_dim1=classifier_config.hidden_dim1,
        hidden_dim2=classifier_config.hidden_dim2,
        dropout_rate=classifier_config.dropout_rate
    )
    
    # Initialize trainer
    trainer = ConfidenceTrainer(
        model=ensemble,
        device=device,
        gamma=classifier_config.gamma,
        learning_rate=1e-3
    )
    
    # Generate synthetic training data
    print(f"Generating synthetic training data...")
    hidden_states_dict, targets_dict = generate_synthetic_training_data(
        num_samples=100,
        num_layers=len(model_config.exit_layer_indices),
        hidden_dim=classifier_config.input_dim,
        device=device
    )
    
    # Training loop
    print(f"\nTraining for 10 epochs...")
    num_epochs = 10
    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(
            hidden_states_dict=hidden_states_dict,
            targets_dict=targets_dict,
            batch_size=32
        )
        
        val_loss, metrics = trainer.eval(
            hidden_states_dict=hidden_states_dict,
            targets_dict=targets_dict
        )
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Print ECE for first layer
        first_layer = model_config.exit_layer_indices[0]
        ece = metrics.get(f'layer_{first_layer}_ece', 0.0)
        print(f"  ECE (Layer {first_layer}): {ece:.4f}")
    
    # Apply temperature scaling
    print(f"\nApplying temperature scaling for calibration...")
    trainer.apply_temperature_scaling(
        hidden_states_dict=hidden_states_dict,
        targets_dict=targets_dict,
        target_ece=classifier_config.target_ece
    )
    
    # Final evaluation
    print(f"\nFinal evaluation after calibration...")
    final_loss, final_metrics = trainer.eval(
        hidden_states_dict=hidden_states_dict,
        targets_dict=targets_dict
    )
    
    print(f"Final Loss: {final_loss:.4f}")
    print("\nECE by layer (target: <0.05):")
    for layer_idx in model_config.exit_layer_indices:
        ece = final_metrics.get(f'layer_{layer_idx}_ece', 0.0)
        print(f"  Layer {layer_idx}: {ece:.4f}")
    
    # Save model
    model_path = "checkpoints/confidence_classifier.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(ensemble.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    return ensemble, trainer


if __name__ == "__main__":
    train_confidence_classifiers()
