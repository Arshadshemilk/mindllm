"""
Phase 2: Meta Confidence Classifier (Offline Training)
Implements confidence estimation MLPs with geometric-like loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional
import numpy as np
from sklearn.calibration import calibration_curve


class ConfidenceHead(nn.Module):
    """
    Lightweight MLP for confidence estimation at a specific exit layer.
    
    Architecture:
    - Input: hidden_dim (2048)
    - Hidden1: 256 units with ReLU
    - Hidden2: 64 units with ReLU
    - Output: 1 unit with Sigmoid
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim1: int = 256,
        hidden_dim2: int = 64,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            
        Returns:
            confidence: [batch_size, seq_len, 1] or [batch_size, 1]
        """
        # Remember original shape for reshaping back
        original_shape = hidden_state.shape[:-1]
        
        # Flatten all but last dimension
        if hidden_state.dim() > 2:
            hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1])
        
        # Forward pass
        x = F.relu(self.dense1(hidden_state))
        x = self.dropout1(x)
        x = F.relu(self.dense2(x))
        x = self.dropout2(x)
        x = self.output(x)
        confidence = self.sigmoid(x)
        
        # Reshape back
        confidence = confidence.reshape(*original_shape, 1)
        return confidence


class ConfidenceClassifierEnsemble(nn.Module):
    """
    Ensemble of confidence heads, one for each exit layer.
    """
    
    def __init__(
        self,
        exit_layer_indices: List[int],
        input_dim: int = 2048,
        hidden_dim1: int = 256,
        hidden_dim2: int = 64,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.exit_layer_indices = sorted(exit_layer_indices)
        
        # Create confidence head for each exit layer
        self.heads = nn.ModuleDict()
        for layer_idx in self.exit_layer_indices:
            self.heads[str(layer_idx)] = ConfidenceHead(
                input_dim=input_dim,
                hidden_dim1=hidden_dim1,
                hidden_dim2=hidden_dim2,
                dropout_rate=dropout_rate
            )
    
    def forward(
        self,
        hidden_states: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """
        Forward pass through all confidence heads.
        
        Args:
            hidden_states: Dict mapping layer index to hidden state tensor
            
        Returns:
            Dict mapping layer index to confidence scores [0, 1]
        """
        confidences = {}
        for layer_idx in self.exit_layer_indices:
            if layer_idx in hidden_states:
                confidences[layer_idx] = self.heads[str(layer_idx)](
                    hidden_states[layer_idx]
                ).squeeze(-1)  # Remove last dimension
        
        return confidences


class GeometricBCELoss(nn.Module):
    """
    Geometric-like loss function with discount factor.
    
    Formula: L_geom = sum_l gamma^(L-l) * BCE^(l)
    where L is the final layer, l is the current layer, and gamma is the discount factor.
    """
    
    def __init__(self, gamma: float = 0.9):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(
        self,
        predictions: Dict[int, torch.Tensor],
        targets: Dict[int, torch.Tensor],
        final_layer_idx: int
    ) -> torch.Tensor:
        """
        Compute geometric loss across all layers.
        
        Args:
            predictions: Dict mapping layer idx to predicted confidences
            targets: Dict mapping layer idx to binary target labels
            final_layer_idx: Index of final/reference layer
            
        Returns:
            Total weighted loss
        """
        total_loss = 0.0
        
        for layer_idx in sorted(predictions.keys()):
            pred = predictions[layer_idx]
            target = targets[layer_idx]
            
            # Compute BCE for this layer
            layer_bce = self.bce(pred, target).mean()
            
            # Compute discount factor
            discount = self.gamma ** (final_layer_idx - layer_idx)
            
            # Add weighted loss
            total_loss = total_loss + discount * layer_bce
        
        return total_loss


class ConfidenceTrainer:
    """
    Trainer for confidence classifiers using geometric loss.
    """
    
    def __init__(
        self,
        model: ConfidenceClassifierEnsemble,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        gamma: float = 0.9,
        learning_rate: float = 1e-3
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = GeometricBCELoss(gamma=gamma)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def generate_targets(
        self,
        exit_predictions: torch.Tensor,
        final_predictions: torch.Tensor,
        threshold: float = 0.9
    ) -> torch.Tensor:
        """
        Generate binary labels by comparing exit layer predictions to final layer.
        
        Args:
            exit_predictions: Logits from exit layer [batch_size, seq_len, vocab_size]
            final_predictions: Logits from final layer [batch_size, seq_len, vocab_size]
            threshold: Similarity threshold for positive label
            
        Returns:
            Binary targets [batch_size, seq_len]
        """
        # Get top-1 predictions
        exit_top1 = exit_predictions.argmax(dim=-1)
        final_top1 = final_predictions.argmax(dim=-1)
        
        # Label as confident (1) if predictions match, else uncertain (0)
        targets = (exit_top1 == final_top1).float()
        
        return targets
    
    def train_epoch(
        self,
        hidden_states_dict: Dict[int, List[torch.Tensor]],
        targets_dict: Dict[int, List[torch.Tensor]],
        batch_size: int = 32
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            hidden_states_dict: Dict mapping layer idx to list of hidden states
            targets_dict: Dict mapping layer idx to list of target labels
            batch_size: Batch size for training
            
        Returns:
            Average loss over the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Get number of samples
        first_layer = list(hidden_states_dict.keys())[0]
        num_samples = len(hidden_states_dict[first_layer])
        
        # Iterate through batches
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            
            # Collect batch hidden states and targets
            batch_hidden_states = {}
            batch_targets = {}
            
            for layer_idx in hidden_states_dict.keys():
                states = [hidden_states_dict[layer_idx][i] for i in range(batch_start, batch_end)]
                batch_hidden_states[layer_idx] = torch.cat(states, dim=0).to(self.device)
                
                targets = [targets_dict[layer_idx][i] for i in range(batch_start, batch_end)]
                batch_targets[layer_idx] = torch.cat(targets, dim=0).to(self.device)
            
            # Forward pass
            predictions = self.model(batch_hidden_states)
            
            # Compute loss
            loss = self.criterion(
                predictions,
                batch_targets,
                final_layer_idx=max(hidden_states_dict.keys())
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    @torch.no_grad()
    def eval(
        self,
        hidden_states_dict: Dict[int, List[torch.Tensor]],
        targets_dict: Dict[int, List[torch.Tensor]]
    ) -> Tuple[float, Dict]:
        """
        Evaluate the model.
        
        Args:
            hidden_states_dict: Dict mapping layer idx to list of hidden states
            targets_dict: Dict mapping layer idx to list of target labels
            
        Returns:
            Average loss and metrics dictionary
        """
        self.model.eval()
        
        # Collect all hidden states and targets
        all_hidden_states = {}
        all_targets = {}
        
        for layer_idx in hidden_states_dict.keys():
            states = hidden_states_dict[layer_idx]
            all_hidden_states[layer_idx] = torch.cat(states, dim=0).to(self.device)
            
            targets = targets_dict[layer_idx]
            all_targets[layer_idx] = torch.cat(targets, dim=0).to(self.device)
        
        # Forward pass
        predictions = self.model(all_hidden_states)
        
        # Compute loss
        loss = self.criterion(
            predictions,
            all_targets,
            final_layer_idx=max(hidden_states_dict.keys())
        )
        
        # Compute calibration metrics
        metrics = {}
        for layer_idx in predictions.keys():
            pred = predictions[layer_idx].cpu().numpy()
            target = all_targets[layer_idx].cpu().numpy().astype(int)
            
            # Expected Calibration Error (ECE)
            ece = self.compute_ece(pred, target)
            metrics[f'layer_{layer_idx}_ece'] = ece
        
        return loss.item(), metrics
    
    @staticmethod
    def compute_ece(
        confidence: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        Args:
            confidence: Predicted confidences [0, 1]
            labels: Binary labels {0, 1}
            n_bins: Number of bins for ECE calculation
            
        Returns:
            ECE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = confidence[in_bin].mean()
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
        
        return ece
    
    def apply_temperature_scaling(
        self,
        hidden_states_dict: Dict[int, List[torch.Tensor]],
        targets_dict: Dict[int, List[torch.Tensor]],
        target_ece: float = 0.05
    ):
        """
        Apply temperature scaling to calibrate confidence estimates.
        
        Args:
            hidden_states_dict: Dict mapping layer idx to list of hidden states
            targets_dict: Dict mapping layer idx to list of target labels
            target_ece: Target ECE value
        """
        self.model.eval()
        
        # Get predictions
        all_hidden_states = {}
        for layer_idx in hidden_states_dict.keys():
            states = hidden_states_dict[layer_idx]
            all_hidden_states[layer_idx] = torch.cat(states, dim=0).to(self.device)
        
        predictions = self.model(all_hidden_states)
        
        # Optimize temperature for each layer
        for layer_idx in predictions.keys():
            pred = predictions[layer_idx].cpu().numpy()
            target = targets_dict[layer_idx]
            all_targets = torch.cat(target, dim=0).cpu().numpy().astype(int)
            
            # Find optimal temperature
            best_temp = 1.0
            best_ece = float('inf')
            
            for temp in np.linspace(0.5, 2.0, 30):
                calibrated = 1.0 / (1.0 + np.exp(-np.log(pred / (1 - pred + 1e-10)) / temp))
                ece = self.compute_ece(calibrated, all_targets)
                if ece < best_ece:
                    best_ece = ece
                    best_temp = temp
            
            # Apply temperature scaling
            head = self.model.heads[str(layer_idx)]
            head.temperature = best_temp
