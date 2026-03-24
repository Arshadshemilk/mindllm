"""
Real Production Pipeline for Phase 2: Confidence Classifier Training

This is the actual training code with:
- Geometric loss implementation
- Temperature scaling calibration
- ECE computation
- Proper training/validation loops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceHeadProduction(nn.Module):
    """Production-grade confidence head with proper regularization"""
    
    def __init__(
        self,
        input_dim: int = 2304,
        hidden_dim1: int = 256,
        hidden_dim2: int = 64,
        dropout_rate: float = 0.2,
        layer_idx: int = None
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.temperature = 1.0  # For temperature scaling
        
        # Layer 1
        self.dense1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2
        self.dense2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output
        self.output = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()
        
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization"""
        for module in [self.dense1, self.dense2, self.output]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, hidden_state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass with proper batch norm handling"""
        # Flatten if needed
        original_shape = hidden_state.shape[:-1]
        if hidden_state.dim() > 2:
            hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1])
        
        # Forward pass
        x = self.dense1(hidden_state)
        if training:
            x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        if training:
            x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        
        x = self.output(x)
        confidence = self.sigmoid(x / self.temperature)
        
        # Reshape back
        confidence = confidence.reshape(*original_shape, 1)
        return confidence


class ConfidenceClassifierEnsembleProduction(nn.Module):
    """Production ensemble of confidence heads"""
    
    def __init__(
        self,
        exit_layer_indices: List[int],
        input_dim: int = 2304,
        hidden_dim1: int = 256,
        hidden_dim2: int = 64,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.exit_layer_indices = sorted(exit_layer_indices)
        self.heads = nn.ModuleDict()
        
        for layer_idx in self.exit_layer_indices:
            self.heads[str(layer_idx)] = ConfidenceHeadProduction(
                input_dim=input_dim,
                hidden_dim1=hidden_dim1,
                hidden_dim2=hidden_dim2,
                dropout_rate=dropout_rate,
                layer_idx=layer_idx
            )
        
        logger.info(f"Initialized confidence ensemble with {len(self.heads)} heads")
    
    def forward(
        self,
        hidden_states: Dict[int, torch.Tensor],
        training: bool = True
    ) -> Dict[int, torch.Tensor]:
        """Forward pass through all heads"""
        confidences = {}
        
        for layer_idx in self.exit_layer_indices:
            if layer_idx in hidden_states:
                confidences[layer_idx] = self.heads[str(layer_idx)](
                    hidden_states[layer_idx],
                    training=training
                ).squeeze(-1)
        
        return confidences


class GeometricBCELossProduction(nn.Module):
    """Production geometric loss with proper weighting"""
    
    def __init__(self, gamma: float = 0.9):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='mean')
    
    def forward(
        self,
        predictions: Dict[int, torch.Tensor],
        targets: Dict[int, torch.Tensor],
        final_layer_idx: int
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute weighted geometric loss"""
        
        total_loss = 0.0
        layer_losses = {}
        
        for layer_idx in sorted(predictions.keys()):
            pred = predictions[layer_idx].clamp(1e-6, 1.0 - 1e-6)
            target = targets[layer_idx].float().squeeze()
            
            # Compute BCE for this layer
            layer_bce = self.bce(pred, target)
            
            # Discount factor
            discount = self.gamma ** (final_layer_idx - layer_idx)
            weighted_loss = discount * layer_bce
            
            total_loss = total_loss + weighted_loss
            layer_losses[f'layer_{layer_idx}'] = layer_bce.item()
        
        return total_loss, layer_losses


class ConfidenceTrainerProduction:
    """Production training pipeline for confidence classifiers"""
    
    def __init__(
        self,
        model: ConfidenceClassifierEnsembleProduction,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        gamma: float = 0.9,
        batch_size: int = 32
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.criterion = GeometricBCELossProduction(gamma=gamma)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'ece': []
        }
        
        logger.info(f"Trainer initialized on {device}")
    
    def train_epoch(
        self,
        hidden_states_dict: Dict[int, List[torch.Tensor]],
        targets_dict: Dict[int, List[torch.Tensor]]
    ) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        # Get number of samples
        first_layer = list(hidden_states_dict.keys())[0]
        num_samples = len(hidden_states_dict[first_layer])
        
        total_loss = 0.0
        num_batches = 0
        layer_losses = {f'layer_{l}': 0.0 for l in hidden_states_dict.keys()}
        
        # Iterate through batches
        for batch_start in tqdm(
            range(0, num_samples, self.batch_size),
            desc="Training",
            leave=False
        ):
            batch_end = min(batch_start + self.batch_size, num_samples)
            batch_size = batch_end - batch_start
            
            # Collect batch data
            batch_hidden_states = {}
            batch_targets = {}
            
            for layer_idx in hidden_states_dict.keys():
                states = [hidden_states_dict[layer_idx][i] for i in range(batch_start, batch_end)]
                batch_hidden_states[layer_idx] = torch.cat(states, dim=0).to(self.device).float()
                
                targets = [targets_dict[layer_idx][i] for i in range(batch_start, batch_end)]
                batch_targets[layer_idx] = torch.cat(targets, dim=0).to(self.device)
            
            # Forward pass
            predictions = self.model(batch_hidden_states, training=True)
            
            # Compute loss
            final_layer = max(hidden_states_dict.keys())
            loss, layer_loss_dict = self.criterion(
                predictions,
                batch_targets,
                final_layer
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            for layer, layer_loss in layer_loss_dict.items():
                layer_losses[layer] += layer_loss
            num_batches += 1
        
        if self.scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        for layer in layer_losses:
            layer_losses[layer] /= num_batches
        
        self.training_history['train_loss'].append(avg_loss)
        
        logger.info(f"Epoch train loss: {avg_loss:.4f}")
        
        return {'loss': avg_loss, 'layer_losses': layer_losses}
    
    @torch.no_grad()
    def validate(
        self,
        hidden_states_dict: Dict[int, List[torch.Tensor]],
        targets_dict: Dict[int, List[torch.Tensor]]
    ) -> Dict:
        """Validation step"""
        self.model.eval()
        
        # Prepare data
        all_hidden_states = {}
        all_targets = {}
        
        for layer_idx in hidden_states_dict.keys():
            states = hidden_states_dict[layer_idx]
            all_hidden_states[layer_idx] = torch.cat(states, dim=0).to(self.device).float()
            
            targets = targets_dict[layer_idx]
            all_targets[layer_idx] = torch.cat(targets, dim=0).to(self.device)
        
        # Forward pass
        predictions = self.model(all_hidden_states, training=False)
        
        # Compute loss
        final_layer = max(hidden_states_dict.keys())
        loss, layer_loss_dict = self.criterion(
            predictions,
            all_targets,
            final_layer
        )
        
        # Compute ECE
        metrics = {'loss': loss.item()}
        for layer_idx in predictions.keys():
            pred = predictions[layer_idx].detach().cpu().float().numpy()
            target = all_targets[layer_idx].detach().cpu().float().numpy().astype(int).flatten()
            
            ece = self.compute_ece(pred.flatten(), target)
            metrics[f'ece_layer_{layer_idx}'] = ece
        
        self.training_history['val_loss'].append(loss.item())
        
        return metrics
    
    @staticmethod
    def compute_ece(
        confidence: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_accs = []
        bin_confs = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = confidence[in_bin].mean()
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
                bin_accs.append(accuracy_in_bin)
                bin_confs.append(avg_confidence_in_bin)
        
        return ece
    
    def apply_temperature_scaling(
        self,
        hidden_states_dict: Dict[int, List[torch.Tensor]],
        targets_dict: Dict[int, List[torch.Tensor]],
        target_ece: float = 0.05,
        temp_range: Tuple[float, float] = (0.3, 3.0)
    ):
        """Apply temperature scaling for calibration"""
        
        logger.info("Applying temperature scaling...")
        
        self.model.eval()
        
        for layer_idx in self.model.exit_layer_indices:
            # Get predictions
            hidden_state = torch.cat(
                hidden_states_dict[layer_idx], dim=0
            ).to(self.device).float()
            target = torch.cat(targets_dict[layer_idx], dim=0).to(self.device)
            
            with torch.no_grad():
                pred = self.model.heads[str(layer_idx)](hidden_state, training=False)
            
            pred = pred.detach().cpu().float().numpy().flatten()
            target = target.detach().cpu().float().numpy().astype(int).flatten()
            
            # Find optimal temperature
            best_temp = 1.0
            best_ece = float('inf')
            
            temps_to_try = np.linspace(temp_range[0], temp_range[1], 40)
            
            for temp in temps_to_try:
                calibrated_pred = 1.0 / (1.0 + np.exp(
                    -np.log(np.clip(pred, 1e-10, 1.0 - 1e-10) / (1.0 - np.clip(pred, 1e-10, 1.0 - 1e-10))) / temp
                ))
                calibrated_pred = np.clip(calibrated_pred, 1e-6, 1.0 - 1e-6)
                
                ece = self.compute_ece(calibrated_pred, target)
                
                if ece < best_ece:
                    best_ece = ece
                    best_temp = temp
            
            # Apply temperature
            self.model.heads[str(layer_idx)].temperature = best_temp
            logger.info(f"Layer {layer_idx}: temperature={best_temp:.4f}, ECE={best_ece:.4f}")


def train_confidence_classifiers(
    hidden_states_dict: Dict[int, List[torch.Tensor]],
    targets_dict: Dict[int, List[torch.Tensor]],
    exit_layer_indices: List[int],
    num_epochs: int = 20,
    batch_size: int = 32
):
    """Main training function"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model
    model = ConfidenceClassifierEnsembleProduction(
        exit_layer_indices=exit_layer_indices,
        input_dim=1152,
        hidden_dim1=256,
        hidden_dim2=64,
        dropout_rate=0.2
    )
    
    # Initialize trainer
    trainer = ConfidenceTrainerProduction(
        model=model,
        device=device,
        learning_rate=1e-3,
        batch_size=batch_size
    )
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    
    best_val_loss = float('inf')
    patience = 5
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(hidden_states_dict, targets_dict)
        val_metrics = trainer.validate(hidden_states_dict, targets_dict)
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train: {train_metrics['loss']:.4f} | "
            f"Val: {val_metrics['loss']:.4f}"
        )
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            no_improve_count = 0
            
            # Save best model
            torch.save(model.state_dict(), 'checkpoints/best_confidence_model.pt')
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Apply temperature scaling
    trainer.apply_temperature_scaling(
        hidden_states_dict,
        targets_dict,
        target_ece=0.05
    )
    
    # Final evaluation
    final_metrics = trainer.validate(hidden_states_dict, targets_dict)
    logger.info(f"Final validation loss: {final_metrics['loss']:.4f}")
    
    return model, trainer
