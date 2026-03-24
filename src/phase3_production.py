"""
Real Production Pipeline for Phase 3: Bandit Controller with Online Learning

This is the actual online learning code with:
- Real arm selection and reward computation
- Dynamic threshold selection
- Convergence monitoring
- Proper statistical tracking
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArmStatistics:
    """Track statistics for an individual arm"""
    
    arm_id: int
    threshold: float
    num_pulls: int = 0
    total_reward: float = 0.0
    success_count: int = 0
    tokens_saved: int = 0
    confidences: List[float] = field(default_factory=list)
    exit_layers: List[int] = field(default_factory=list)
    
    @property
    def mean_reward(self) -> float:
        """Average reward for this arm"""
        return self.total_reward / max(1, self.num_pulls)
    
    @property
    def success_rate(self) -> float:
        """Success rate (confidences above threshold)"""
        return self.success_count / max(1, self.num_pulls)
    
    @property
    def avg_efficiency(self) -> float:
        """Average efficiency (tokens saved / total)"""
        if self.num_pulls == 0:
            return 0.0
        return self.tokens_saved / (self.num_pulls * 26)  # 26 layers max
    
    def update(
        self,
        confidence: float,
        exit_layer: int,
        early_exit: bool,
        target_reward: float
    ):
        """Update statistics with new sample"""
        self.num_pulls += 1
        self.confidences.append(confidence)
        self.exit_layers.append(exit_layer)
        
        # Reward: confidence vs efficiency tradeoff
        # 0.7 * confidence_metric - 0.3 * depth_penalty
        efficiency = 1.0 if early_exit else 0.0
        reward = 0.7 * (1.0 if confidence >= self.threshold else 0.0) + 0.3 * efficiency
        
        self.total_reward += reward
        
        if confidence >= self.threshold:
            self.success_count += 1
            if early_exit:
                self.tokens_saved += max(0, 26 - exit_layer)


class UCBBanditArm:
    """Single arm in the UCB bandit"""
    
    def __init__(self, arm_id: int, threshold: float):
        self.arm_id = arm_id
        self.threshold = threshold
        self.stats = ArmStatistics(arm_id=arm_id, threshold=threshold)
        self.ucb_value = np.inf  # Initial optimism
    
    def compute_ucb(self, total_pulls: int, exploration_factor: float = 1.41):
        """Compute UCB value with proper confidence bounds"""
        
        if self.stats.num_pulls == 0:
            return np.inf
        
        exploitation = self.stats.mean_reward
        exploration = exploration_factor * np.sqrt(
            np.log(total_pulls) / self.stats.num_pulls
        )
        
        self.ucb_value = exploitation + exploration
        return self.ucb_value
    
    def update(
        self,
        confidence: float,
        exit_layer: int,
        early_exit: bool
    ):
        """Update arm with new sample"""
        # Compute target reward
        efficiency = 1.0 if early_exit else 0.0
        target_reward = 0.7 * (1.0 if confidence >= self.threshold else 0.0) + 0.3 * efficiency
        
        self.stats.update(
            confidence=confidence,
            exit_layer=exit_layer,
            early_exit=early_exit,
            target_reward=target_reward
        )


class ExplorationSchedule:
    """Manage exploration vs exploitation scheduling"""
    
    def __init__(
        self,
        warmup_steps: int = 100,
        high_explore_steps: int = 400,
        std_steps: int = float('inf')
    ):
        self.warmup_steps = warmup_steps
        self.high_explore_steps = high_explore_steps
        self.std_steps = std_steps
        self.current_step = 0
    
    def get_exploration_factor(self) -> float:
        """Get exploration factor based on current phase"""
        
        if self.current_step < self.warmup_steps:
            # Warmup: high exploration, uniform distribution bias
            return 3.0
        elif self.current_step < self.warmup_steps + self.high_explore_steps:
            # Extended exploration phase
            return 2.0
        else:
            # Standard UCB phase
            return 1.41
    
    def step(self):
        """Move to next step"""
        self.current_step += 1
    
    @property
    def phase(self) -> str:
        """Current exploration phase"""
        if self.current_step < self.warmup_steps:
            return "warmup"
        elif self.current_step < self.warmup_steps + self.high_explore_steps:
            return "high_explore"
        else:
            return "standard"


class UCBBanditControllerProduction:
    """Production-grade UCB bandit controller with online learning"""
    
    def __init__(
        self,
        num_arms: int = 20,
        threshold_range: Tuple[float, float] = (0.50, 0.99),
        warmup_steps: int = 100,
        high_explore_steps: int = 400
    ):
        self.num_arms = num_arms
        self.threshold_range = threshold_range
        
        # Create arms with linearly spaced thresholds
        self.thresholds = np.linspace(
            threshold_range[0],
            threshold_range[1],
            num_arms
        )
        
        self.arms = [
            UCBBanditArm(i, float(threshold))
            for i, threshold in enumerate(self.thresholds)
        ]
        
        self.schedule = ExplorationSchedule(
            warmup_steps=warmup_steps,
            high_explore_steps=high_explore_steps
        )
        
        self.metrics = {
            'total_steps': 0,
            'total_tokens': 0,
            'total_early_exits': 0,
            'arm_selections': [0] * num_arms,
            'cumulative_rewards': [0.0] * num_arms,
            'phase_changes': []
        }
        
        logger.info(
            f"Initialized UCB bandit with {num_arms} arms, "
            f"thresholds={threshold_range[0]:.2f} to {threshold_range[1]:.2f}"
        )
    
    def select_arm(self) -> int:
        """Select arm based on UCB strategy"""
        
        total_pulls = sum(arm.stats.num_pulls for arm in self.arms)
        exploration_factor = self.schedule.get_exploration_factor()
        
        # Compute UCB for all arms
        ucb_values = []
        for arm in self.arms:
            ucb = arm.compute_ucb(max(1, total_pulls), exploration_factor)
            ucb_values.append(ucb)
        
        # Select best arm
        best_arm_idx = np.argmax(ucb_values)
        
        return best_arm_idx
    
    def pull_arm(
        self,
        arm_idx: int,
        confidence: float,
        exit_layer: int
    ) -> Dict:
        """Pull selected arm and update statistics"""
        
        arm = self.arms[arm_idx]
        threshold = arm.threshold
        
        # Determine if early exit occurred
        early_exit = confidence >= threshold
        
        # Update arm statistics
        arm.update(
            confidence=confidence,
            exit_layer=exit_layer,
            early_exit=early_exit
        )
        
        # Update global metrics
        self.metrics['total_steps'] += 1
        self.metrics['total_tokens'] += 1
        self.metrics['arm_selections'][arm_idx] += 1
        self.metrics['cumulative_rewards'][arm_idx] += arm.stats.mean_reward
        
        if early_exit:
            self.metrics['total_early_exits'] += 1
            tokens_saved = max(0, 26 - exit_layer)
            self.metrics['total_tokens'] -= tokens_saved
        
        # Check for phase change
        current_phase = self.schedule.phase
        self.schedule.step()
        new_phase = self.schedule.phase
        
        if current_phase != new_phase:
            logger.info(f"Phase change: {current_phase} → {new_phase}")
            self.metrics['phase_changes'].append(
                (self.metrics['total_steps'], new_phase)
            )
        
        return {
            'arm_id': arm_idx,
            'threshold': threshold,
            'early_exit': early_exit,
            'reward': arm.stats.mean_reward,
            'phase': self.schedule.phase,
            'ucb_value': arm.ucb_value
        }
    
    def get_best_arm(self) -> Tuple[int, float]:
        """Get best arm based on empirical mean"""
        best_idx = int(np.argmax(
            [arm.stats.mean_reward for arm in self.arms]
        ))
        return best_idx, self.arms[best_idx].threshold
    
    def get_arm_statistics(self) -> Dict:
        """Get comprehensive arm statistics"""
        stats = {}
        
        for i, arm in enumerate(self.arms):
            stats[f'arm_{i}'] = {
                'threshold': arm.threshold,
                'num_pulls': arm.stats.num_pulls,
                'mean_reward': arm.stats.mean_reward,
                'success_rate': arm.stats.success_rate,
                'avg_efficiency': arm.stats.avg_efficiency,
                'ucb_value': arm.ucb_value
            }
        
        return stats
    
    def get_convergence_metrics(self) -> Dict:
        """Compute convergence metrics"""
        
        arm_rewards = [arm.stats.mean_reward for arm in self.arms]
        
        best_idx = int(np.argmax(arm_rewards))
        best_reward = arm_rewards[best_idx]
        
        # Regret: how much worse than optimal
        total_regret = 0.0
        for i, arm in enumerate(self.arms):
            regret_per_sample = best_reward - arm.stats.mean_reward
            total_regret += arm.stats.num_pulls * regret_per_sample
        
        # Arm concentration: how much exploitation vs exploration
        total_pulls = sum(arm.stats.num_pulls for arm in self.arms)
        best_arm_pulls = self.arms[best_idx].stats.num_pulls
        concentration = best_arm_pulls / max(1, total_pulls)
        
        return {
            'best_arm': best_idx,
            'best_threshold': self.arms[best_idx].threshold,
            'best_reward': best_reward,
            'total_regret': total_regret,
            'avg_regret': total_regret / max(1, total_pulls),
            'arm_concentration': concentration,
            'exploration_phase': self.schedule.phase
        }
    
    def log_statistics(self):
        """Log detailed statistics"""
        
        logger.info("\n" + "="*60)
        logger.info("BANDIT STATISTICS")
        logger.info("="*60)
        
        # Global metrics
        logger.info(f"Total steps: {self.metrics['total_steps']}")
        logger.info(f"Early exits: {self.metrics['total_early_exits']}")
        early_exit_rate = (
            self.metrics['total_early_exits'] / max(1, self.metrics['total_steps'])
        )
        logger.info(f"Early exit rate: {early_exit_rate:.2%}")
        
        # Convergence metrics
        convergence = self.get_convergence_metrics()
        logger.info(f"Best arm: {convergence['best_arm']} (threshold={convergence['best_threshold']:.4f})")
        logger.info(f"Best reward: {convergence['best_reward']:.4f}")
        logger.info(f"Avg regret: {convergence['avg_regret']:.4f}")
        logger.info(f"Arm concentration: {convergence['arm_concentration']:.2%}")
        
        # Top performing arms
        arm_stats = self.get_arm_statistics()
        arm_rewards = [
            (i, arm_stats[f'arm_{i}']['mean_reward'])
            for i in range(self.num_arms)
        ]
        arm_rewards.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("\nTop 5 arms:")
        for rank, (arm_id, reward) in enumerate(arm_rewards[:5], 1):
            arm = self.arms[arm_id]
            logger.info(
                f"  {rank}. Arm {arm_id} (T={arm.threshold:.4f}): "
                f"reward={reward:.4f}, pulls={arm.stats.num_pulls}, "
                f"success_rate={arm.stats.success_rate:.2%}"
            )
        
        logger.info("="*60 + "\n")


class AdaptiveThresholdManagerProduction:
    """Integration of bandit controller for production inference"""
    
    def __init__(
        self,
        num_arms: int = 20,
        threshold_range: Tuple[float, float] = (0.50, 0.99),
        cache_best_arm: bool = True
    ):
        self.controller = UCBBanditControllerProduction(
            num_arms=num_arms,
            threshold_range=threshold_range
        )
        self.cache_best_arm = cache_best_arm
        self.current_best_arm = None
        self.step_count = 0
        self.cache_update_frequency = 50  # Update cache every N steps
    
    def get_threshold(self) -> float:
        """Get next threshold to use"""
        arm_idx = self.controller.select_arm()
        return self.controller.arms[arm_idx].threshold
    
    def update_with_feedback(
        self,
        confidence: float,
        exit_layer: int,
        early_exit: bool,
        threshold_used: float
    ):
        """Update bandit with inference feedback"""
        
        # Find which arm was used
        arm_idx = int(np.argmin(
            np.abs(self.controller.thresholds - threshold_used)
        ))
        
        # Pull the arm
        result = self.controller.pull_arm(
            arm_idx=arm_idx,
            confidence=confidence,
            exit_layer=exit_layer
        )
        
        self.step_count += 1
        
        # Update cached best arm
        if self.cache_best_arm and self.step_count % self.cache_update_frequency == 0:
            best_idx, _ = self.controller.get_best_arm()
            self.current_best_arm = best_idx
    
    def get_summary(self) -> Dict:
        """Get convergence summary"""
        return self.controller.get_convergence_metrics()


def run_online_learning_episode(
    controller: UCBBanditControllerProduction,
    confidences: List[float],
    exit_layers: List[int],
    num_episodes: int = 1
) -> Dict:
    """
    Simulate online learning episode with real confidences and exit patterns
    
    Args:
        controller: The bandit controller
        confidences: List of confidence values across samples
        exit_layers: List of exit layers for samples
        num_episodes: Number of full episodes to run
    
    Returns:
        Dictionary with learning metrics
    """
    
    results = {
        'episode_rewards': [],
        'episode_early_exits': [],
        'arm_stats': []
    }
    
    num_samples = len(confidences)
    
    for episode in range(num_episodes):
        episode_reward = 0.0
        episode_early_exits = 0
        
        for i in range(num_samples):
            # Select arm
            arm_idx = controller.select_arm()
            
            # Pull arm with real data
            result = controller.pull_arm(
                arm_idx=arm_idx,
                confidence=confidences[i],
                exit_layer=exit_layers[i]
            )
            
            episode_reward += result['reward']
            if result['early_exit']:
                episode_early_exits += 1
        
        results['episode_rewards'].append(episode_reward / num_samples)
        results['episode_early_exits'].append(episode_early_exits)
        
        logger.info(
            f"Episode {episode+1}: avg_reward={results['episode_rewards'][-1]:.4f}, "
            f"early_exits={episode_early_exits}/{num_samples}"
        )
    
    # Final statistics
    results['arm_stats'] = controller.get_arm_statistics()
    results['convergence'] = controller.get_convergence_metrics()
    
    return results
