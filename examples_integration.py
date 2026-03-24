"""
Example script for complete integration of all three phases.

This script demonstrates the full MindEdgeAI pipeline:
1. Loading or creating a multi-exit model
2. Setting up confidence classifiers
3. Initializing the bandit controller
4. Running integrated generation with adaptive thresholds
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.phase1_exit_architecture.multi_exit_llm import MultiExitLLM
from src.phase2_confidence_classifier.confidence_heads import ConfidenceClassifierEnsemble
from src.phase3_bandit_controller.ucb_controller import UCBBanditController
from src.integration import MindEdgeAIEngine
from src.config import ModelConfig, ConfidenceClassifierConfig, BanditConfig
from src.utils.utilities import set_seed


def create_gemma_model_with_unsloth(model_config: ModelConfig, device: str):
    """
    Load Gemma 3 1B model directly from Hugging Face.
    
    Uses unsloth/gemma-3-1b-it which is pre-optimized and ready to use.
    """
    from src.utils.utilities import load_model_with_unsloth
    
    print("Loading Gemma 3 1B (unsloth/gemma-3-1b-it)...")
    
    try:
        base_model, tokenizer = load_model_with_unsloth(model_config.model_name)
        print(f"✓ Model loaded successfully: {model_config.model_name}")
        
        # Wrap with multi-exit architecture
        multi_exit_llm = MultiExitLLM(
            base_model=base_model,
            hidden_dim=model_config.hidden_dim,
            exit_layer_indices=model_config.exit_layer_indices,
            vocab_size=tokenizer.vocab_size
        )
        
        return multi_exit_llm, tokenizer
        
    except Exception as e:
        print(f"Note: {e}")
        print("Creating fallback dummy model for demonstration...")
        
        # Fallback to dummy model
        class DummyModel(torch.nn.Module):
            def __init__(self, hidden_dim=2304, num_layers=26):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                
                # Create dummy layers
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
                ])
            
            def forward(self, input_ids):
                batch_size, seq_len = input_ids.shape
                hidden = torch.randn(batch_size, seq_len, self.hidden_dim, device=input_ids.device)
                return hidden
            
            def get_input_embeddings(self):
                return torch.nn.Embedding(32000, self.hidden_dim)
        
        base_model = DummyModel(
            hidden_dim=model_config.hidden_dim,
            num_layers=model_config.num_layers
        ).to(device)
        
        multi_exit_llm = MultiExitLLM(
            base_model=base_model,
            hidden_dim=model_config.hidden_dim,
            exit_layer_indices=model_config.exit_layer_indices,
            vocab_size=32000
        )
        
        return multi_exit_llm, None


def run_integration_demo():
    """Run the full integration demonstration."""
    print("=" * 70)
    print("MindEdgeAI Integration Demo - All Three Phases")
    print("=" * 70)
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")
    
    # Configuration
    model_config = ModelConfig()
    classifier_config = ConfidenceClassifierConfig()
    bandit_config = BanditConfig()
    
    print("Configuration Summary:")
    print(f"  Model: Gemma 3 1B optimized with Unsloth ({model_config.num_layers} layers, {model_config.hidden_dim}D hidden)")
    print(f"  Exit layers: {model_config.exit_layer_indices}")
    print(f"  Confidence heads: {len(model_config.exit_layer_indices)}")
    print(f"  Bandit arms: {bandit_config.num_arms} thresholds")
    print(f"  Threshold range: [{bandit_config.min_threshold}, {bandit_config.max_threshold}]")
    
    # Phase 1: Create multi-exit model
    print("\n" + "=" * 70)
    print("PHASE 1: Multi-Exit Architecture Setup")
    print("=" * 70)
    multi_exit_llm, tokenizer = create_gemma_model_with_unsloth(model_config, device)
    print(f"✓ Multi-exit LLM initialized with {len(model_config.exit_layer_indices)} exit points")
    
    # Phase 2: Create confidence classifiers
    print("\n" + "=" * 70)
    print("PHASE 2: Confidence Classifier Setup")
    print("=" * 70)
    confidence_classifier = ConfidenceClassifierEnsemble(
        exit_layer_indices=model_config.exit_layer_indices,
        input_dim=classifier_config.input_dim,
        hidden_dim1=classifier_config.hidden_dim1,
        hidden_dim2=classifier_config.hidden_dim2,
        dropout_rate=classifier_config.dropout_rate
    )
    print(f"✓ Confidence classifier ensemble initialized")
    print(f"  Heads: {len(confidence_classifier.heads)}")
    print(f"  Architecture: {classifier_config.input_dim}→{classifier_config.hidden_dim1}→" 
          f"{classifier_config.hidden_dim2}→1")
    print(f"  Target ECE: {classifier_config.target_ece}")
    
    # Phase 3: Create bandit controller
    print("\n" + "=" * 70)
    print("PHASE 3: Bandit Controller Setup")
    print("=" * 70)
    bandit_controller = UCBBanditController(
        min_threshold=bandit_config.min_threshold,
        max_threshold=bandit_config.max_threshold,
        num_arms=bandit_config.num_arms,
        exploration_constant_warmup=bandit_config.exploration_constant_warmup,
        exploration_constant_high=bandit_config.exploration_constant_high,
        exploration_constant_standard=bandit_config.exploration_constant_standard,
        confidence_weight=bandit_config.confidence_weight,
        energy_weight=bandit_config.energy_weight,
        warmup_tokens=bandit_config.warmup_tokens,
        extended_explore_tokens=bandit_config.extended_explore_tokens
    )
    print(f"✓ UCB Bandit controller initialized")
    print(f"  Arms: {bandit_controller.num_arms}")
    print(f"  Exploration phases:")
    print(f"    - Warmup (0-{bandit_config.warmup_tokens}): Round-robin")
    print(f"    - Extended (0-{bandit_config.extended_explore_tokens}): High exploration (c={bandit_config.exploration_constant_high})")
    print(f"    - Standard ({bandit_config.extended_explore_tokens}+): Standard (c={bandit_config.exploration_constant_standard})")
    
    # Integration
    print("\n" + "=" * 70)
    print("INTEGRATION: Creating MindEdgeAI Engine")
    print("=" * 70)
    engine = MindEdgeAIEngine(
        multi_exit_model=multi_exit_llm,
        confidence_classifier=confidence_classifier,
        bandit_controller=bandit_controller,
        max_layers=model_config.num_layers,
        device=device
    )
    print(f"✓ MindEdgeAI Engine initialized successfully")
    
    # Generate sample
    print("\n" + "=" * 70)
    print("GENERATION: Running Single Token Generation")
    print("=" * 70)
    
    # Create dummy input
    input_ids = torch.randint(0, 32000, (1, 10), device=device)
    print(f"Input shape: {input_ids.shape}")
    
    try:
        # Generate a single token
        result = engine.generate_token_with_adaptive_threshold(
            input_ids=input_ids,
            token_position=0,
            energy_budget_remaining=1.0,
            use_confidence_classifiers=True
        )
        
        print(f"\n✓ Token generated successfully")
        print(f"  Exit layer: {result['exit_layer']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Threshold used: {result['threshold_used']:.4f}")
        print(f"  Token energy: {result['token_energy']:.4f}")
        print(f"  Selected arm: {result['arm_idx']}")
        
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        print("(This is expected with dummy model - in production uses real Gemma-1B)")
    
    # Show system configuration
    print("\n" + "=" * 70)
    print("SYSTEM CONFIGURATION")
    print("=" * 70)
    config = engine.get_model_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Show current statistics
    print("\n" + "=" * 70)
    print("SYSTEM STATISTICS")
    print("=" * 70)
    stats = engine.get_statistics()
    
    print(f"\nMetrics:")
    for key, value in stats['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    print(f"\nBandit Statistics:")
    for key, value in stats['bandit_stats'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Integration Demo Complete!")
    print("=" * 70)
    
    print("\nNext Steps for Deployment:")
    print("1. Load real Gemma-1B model for Phase 1")
    print("2. Train confidence classifiers (Phase 2) on actual forward passes")
    print("3. Fine-tune bandit controller during initial inference runs")
    print("4. Monitor energy consumption and adjust weights as needed")


if __name__ == "__main__":
    run_integration_demo()
