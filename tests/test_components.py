"""
Unit tests for MindEdgeAI components.

Tests cover:
- Phase 1: Multi-exit architecture
- Phase 2: Confidence classifiers
- Phase 3: Bandit controller
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from src.config import ModelConfig, ConfidenceClassifierConfig, BanditConfig
from src.phase1_exit_architecture.multi_exit_llm import ExitHead, MultiExitLLM
from src.phase2_confidence_classifier.confidence_heads import (
    ConfidenceHead, ConfidenceClassifierEnsemble, GeometricBCELoss
)
from src.phase3_bandit_controller.ucb_controller import (
    BanditArm, UCBBanditController, BanditState
)
from src.utils.utilities import EnergyEstimator, MetricsTracker


class TestExitHead:
    """Test Phase 1 exit head components."""
    
    def test_exit_head_forward(self):
        """Test exit head forward pass."""
        hidden_dim = 2048
        vocab_size = 32000
        batch_size = 2
        seq_len = 10
        
        head = ExitHead(hidden_dim, vocab_size)
        hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
        
        logits = head(hidden_state)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert not torch.isnan(logits).any()
    
    def test_exit_head_backward(self):
        """Test exit head gradient computation."""
        hidden_dim = 2048
        vocab_size = 32000
        batch_size = 2
        seq_len = 10
        
        head = ExitHead(hidden_dim, vocab_size)
        hidden_state = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        
        logits = head(hidden_state)
        loss = logits.sum()
        loss.backward()
        
        assert hidden_state.grad is not None
        assert torch.any(hidden_state.grad != 0)


class TestConfidenceHead:
    """Test Phase 2 confidence head components."""
    
    def test_confidence_head_forward(self):
        """Test confidence head produces valid probabilities."""
        config = ConfidenceClassifierConfig()
        head = ConfidenceHead(
            input_dim=config.input_dim,
            hidden_dim1=config.hidden_dim1,
            hidden_dim2=config.hidden_dim2
        )
        
        hidden_state = torch.randn(4, 32, config.input_dim)
        confidence = head(hidden_state)
        
        # Check output shape
        assert confidence.shape == (4, 32, 1)
        
        # Check values are in valid range
        assert torch.all(confidence >= 0.0)
        assert torch.all(confidence <= 1.0)
    
    def test_confidence_classifier_ensemble(self):
        """Test confidence classifier ensemble."""
        config = ConfidenceClassifierConfig()
        model_config = ModelConfig()
        
        ensemble = ConfidenceClassifierEnsemble(
            exit_layer_indices=model_config.exit_layer_indices,
            input_dim=config.input_dim
        )
        
        hidden_states = {
            layer_idx: torch.randn(4, 32, config.input_dim)
            for layer_idx in model_config.exit_layer_indices
        }
        
        confidences = ensemble(hidden_states)
        
        # Check all layers have outputs
        assert len(confidences) == len(model_config.exit_layer_indices)
        
        # Check output shapes and ranges
        for layer_idx, conf in confidences.items():
            assert conf.shape == (4, 32)
            assert torch.all(conf >= 0.0)
            assert torch.all(conf <= 1.0)
    
    def test_geometric_bce_loss(self):
        """Test geometric loss computation."""
        loss_fn = GeometricBCELoss(gamma=0.9)
        
        predictions = {
            4: torch.rand(8, 32),
            8: torch.rand(8, 32),
            12: torch.rand(8, 32),
            18: torch.rand(8, 32)
        }
        
        targets = {
            4: torch.randint(0, 2, (8, 32)).float(),
            8: torch.randint(0, 2, (8, 32)).float(),
            12: torch.randint(0, 2, (8, 32)).float(),
            18: torch.randint(0, 2, (8, 32)).float()
        }
        
        loss = loss_fn(predictions, targets, final_layer_idx=18)
        
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)


class TestBanditController:
    """Test Phase 3 bandit controller components."""
    
    def test_bandit_arm(self):
        """Test individual bandit arm."""
        arm = BanditArm(arm_index=0, threshold=0.5)
        
        # Simulate updates
        rewards = [0.5, 0.6, 0.55]
        for r in rewards:
            arm.update(r)
        
        assert arm.n_pulls == len(rewards)
        assert abs(arm.mean_reward - np.mean(rewards)) < 1e-6
    
    def test_bandit_ucb_computation(self):
        """Test UCB value computation."""
        arm = BanditArm(arm_index=0, threshold=0.5)
        
        # Before any pulls, UCB should be infinite
        assert arm.get_ucb(1.0) == float('inf')
        
        # After pulls, UCB should be finite
        for _ in range(5):
            arm.update(0.5)
        
        ucb = arm.get_ucb(1.0)
        assert ucb < float('inf')
        assert ucb >= arm.mean_reward  # UCB always >= mean
    
    def test_ucb_bandit_initialization(self):
        """Test UCB bandit controller initialization."""
        config = BanditConfig()
        
        bandit = UCBBanditController(
            min_threshold=config.min_threshold,
            max_threshold=config.max_threshold,
            num_arms=config.num_arms
        )
        
        assert len(bandit.arms) == config.num_arms
        
        # Check thresholds are properly distributed
        thresholds = [arm.threshold for arm in bandit.arms]
        assert thresholds[0] == config.min_threshold
        assert thresholds[-1] == config.max_threshold
        assert all(thresholds[i] < thresholds[i+1] for i in range(len(thresholds)-1))
    
    def test_bandit_threshold_selection(self):
        """Test threshold selection."""
        config = BanditConfig()
        bandit = UCBBanditController(num_arms=config.num_arms)
        
        # Warmup phase: forced round-robin
        for i in range(config.warmup_tokens):
            state = BanditState(token_position=i)
            threshold, arm_idx = bandit.select_threshold(state)
            expected_arm = i % config.num_arms
            assert arm_idx == expected_arm
    
    def test_bandit_reward_computation(self):
        """Test reward function."""
        bandit = UCBBanditController()
        
        # Test reward at early exit (low energy, high confidence)
        reward_early = bandit.compute_reward(exit_layer=4, exit_confidence=0.95)
        
        # Test reward at late exit (high energy, same confidence)
        reward_late = bandit.compute_reward(exit_layer=18, exit_confidence=0.95)
        
        # Early exit should have higher reward (less energy cost)
        assert reward_early > reward_late
    
    def test_bandit_state_computation(self):
        """Test system state computation."""
        bandit = UCBBanditController()
        
        # Add some history
        for i in range(10):
            state = bandit.get_system_state(token_position=i)
            bandit.update_arm(
                arm_idx=i % bandit.num_arms,
                exit_layer=10,
                exit_confidence=0.7 + i * 0.01
            )
        
        # Check state computation
        final_state = bandit.get_system_state(token_position=10)
        assert final_state.total_pulls == 10
        assert final_state.token_position == 10
        assert 0 <= final_state.historical_confidence_mean <= 1


class TestUtilities:
    """Test utility functions."""
    
    def test_energy_estimator(self):
        """Test energy estimation."""
        estimator = EnergyEstimator(max_layers=18)
        
        # Early exit should use less energy
        energy_early = estimator.estimate_token_energy(exit_layer=4)
        energy_late = estimator.estimate_token_energy(exit_layer=18)
        
        assert energy_early < energy_late
        assert energy_early > 0
        assert energy_late <= 1.0
    
    def test_metrics_tracker(self):
        """Test metrics tracking."""
        tracker = MetricsTracker()
        
        # Record some metrics
        tracker.record('loss', 0.5)
        tracker.record('loss', 0.4)
        tracker.record('loss', 0.3)
        tracker.record_confidence(0.8)
        tracker.record_exit_layer(10)
        
        summary = tracker.get_summary()
        
        assert 'loss_mean' in summary
        assert 'confidence_mean' in summary
        assert 'exit_layer_mean' in summary
        assert abs(summary['loss_mean'] - 0.4) < 1e-6
        assert summary['confidence_mean'] == 0.8


class TestConfiguration:
    """Test configuration classes."""
    
    def test_model_config(self):
        """Test model configuration."""
        config = ModelConfig()
        
        assert config.num_layers == 18
        assert config.hidden_dim == 2048
        assert len(config.exit_layer_indices) == 8
        assert config.exit_layer_indices == [4, 6, 8, 10, 12, 14, 16, 18]
    
    def test_classifier_config(self):
        """Test classifier configuration."""
        config = ConfidenceClassifierConfig()
        
        assert config.input_dim == 2048
        assert config.hidden_dim1 == 256
        assert config.hidden_dim2 == 64
        assert config.gamma == 0.9
        assert config.target_ece == 0.05
    
    def test_bandit_config(self):
        """Test bandit configuration."""
        config = BanditConfig()
        
        assert config.min_threshold == 0.50
        assert config.max_threshold == 0.99
        assert config.num_arms == 20
        
        # Test threshold mapping
        threshold_0 = config.get_threshold_at_arm(0)
        threshold_19 = config.get_threshold_at_arm(19)
        
        assert threshold_0 == 0.50
        assert threshold_19 == 0.99


def run_all_tests():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == "__main__":
    run_all_tests()
