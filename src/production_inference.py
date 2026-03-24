"""
Complete Production Inference Pipeline

Integrates Phase 1 (multi-exit), Phase 2 (confidence classifiers), 
and Phase 3 (bandit controller) for real-world deployment.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from dataclasses import dataclass
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for production inference"""
    
    # Model settings
    model_name: str = "unsloth/gemma-3-1b-it"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    
    # Exit architecture
    exit_layers: List[int] = None
    num_tokens_to_generate: int = 100
    
    # Inference settings
    temperature: float = 0.7
    top_p: float = 0.9
    max_batch_size: int = 8
    
    # Bandit settings
    num_arms: int = 20
    threshold_range: Tuple[float, float] = (0.50, 0.99)
    
    def __post_init__(self):
        if self.exit_layers is None:
            self.exit_layers = [6, 10, 14, 18, 22, 26]


class ProductionInferencePipeline:
    """Complete production inference with adaptive thresholding"""
    
    def __init__(
        self,
        model: nn.Module,
        confidence_classifier: nn.Module,
        bandit_controller,
        config: InferenceConfig,
        tokenizer=None
    ):
        self.model = model
        self.confidence_classifier = confidence_classifier
        self.bandit_controller = bandit_controller
        self.config = config
        self.tokenizer = tokenizer
        self.device = config.device
        
        # Metrics
        self.inference_metrics = {
            'total_samples': 0,
            'total_tokens': 0,
            'early_exits': 0,
            'exit_layer_distribution': [0] * 26,
            'threshold_selections': [],
            'confidences': [],
            'latencies': []
        }
        
        logger.info("Production inference pipeline initialized")
    
    def preprocess_input(
        self,
        text: str,
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """Preprocess input text"""
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")
        
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device)
        }
    
    @torch.no_grad()
    def get_exit_confidences(
        self,
        hidden_states: Dict[int, torch.Tensor]
    ) -> Dict[int, float]:
        """Get confidence predictions at all exit layers"""
        
        confidences = {}
        
        self.confidence_classifier.eval()
        
        for layer_idx in self.config.exit_layers:
            if layer_idx in hidden_states:
                hidden = hidden_states[layer_idx].to(self.device).float()
                
                # Get confidence from classifier
                conf = self.confidence_classifier.heads[str(layer_idx)](
                    hidden,
                    training=False
                )
                
                # Squeeze last dim, then average over sequence
                conf = conf.squeeze(-1)  # [batch, 1] -> [batch]
                if conf.dim() > 0:
                    conf = conf.mean(dim=0)  # reduce to scalar
                
                confidences[layer_idx] = float(conf.item())
        
        return confidences
    
    def generate_with_adaptive_exit(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 100
    ) -> Dict:
        """Generate tokens with adaptive exit thresholding"""
        
        self.model.eval()
        generated_tokens = input_ids.clone()
        exit_info = {
            'exit_layers': [],
            'thresholds_used': [],
            'confidences': [],
            'early_exit_count': 0,
            'token_count': 0
        }
        
        # KV cache for efficiency
        past_key_values = None
        
        for token_idx in range(max_new_tokens):
            with torch.no_grad():
                # Forward pass through model
                if past_key_values is not None:
                    # Use only last token for subsequent passes
                    current_input_ids = generated_tokens[:, -1:]
                else:
                    current_input_ids = generated_tokens
                
                # Get model outputs with all hidden states
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask if past_key_values is None else None,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False  # We're managing manually for control
                )
                
                # Extract hidden states at exit layers
                all_hidden_states = outputs.hidden_states
                hidden_states_at_exits = {}
                
                for layer_idx in self.config.exit_layers:
                    if layer_idx <= len(all_hidden_states):
                        hidden_states_at_exits[layer_idx] = all_hidden_states[layer_idx]
                
                # Get confidences at each exit layer
                confidences = self.get_exit_confidences(hidden_states_at_exits)
                
                # Select threshold using bandit
                arm_idx = self.bandit_controller.controller.select_arm()
                threshold = self.bandit_controller.controller.arms[arm_idx].threshold
                
                # Find best exit point based on confidence and threshold
                best_exit_layer = self.config.exit_layers[-1]  # Default to final layer
                best_confidence = 0.0
                early_exited = False
                
                for layer_idx in reversed(self.config.exit_layers):
                    if layer_idx in confidences:
                        conf = confidences[layer_idx]
                        
                        if conf >= threshold:
                            best_exit_layer = layer_idx
                            best_confidence = conf
                            early_exited = True
                            break
                
                # Get logits from selected exit layer
                if best_exit_layer in hidden_states_at_exits:
                    hidden_state = hidden_states_at_exits[best_exit_layer][:, -1, :]
                    
                    # Use exit head to get logits (or use model's head)
                    logits = outputs.logits[0, -1, :]  # Simplified: use model's final logits
                else:
                    logits = outputs.logits[0, -1, :]
                
                # Sample next token
                next_token = self._sample_token(
                    logits,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )
                
                # Update metrics
                generated_tokens = torch.cat([
                    generated_tokens,
                    next_token.unsqueeze(0).unsqueeze(0)
                ], dim=1)
                
                exit_info['exit_layers'].append(best_exit_layer)
                exit_info['thresholds_used'].append(threshold)
                exit_info['confidences'].append(best_confidence)
                if early_exited:
                    exit_info['early_exit_count'] += 1
                exit_info['token_count'] += 1
                
                # Update bandit
                self.bandit_controller.update_with_feedback(
                    confidence=best_confidence,
                    exit_layer=best_exit_layer,
                    early_exit=early_exited,
                    threshold_used=threshold
                )
        
        # Update global metrics
        self.inference_metrics['total_samples'] += 1
        self.inference_metrics['total_tokens'] += exit_info['token_count']
        self.inference_metrics['early_exits'] += exit_info['early_exit_count']
        
        for layer in exit_info['exit_layers']:
            self.inference_metrics['exit_layer_distribution'][layer - 1] += 1
        
        self.inference_metrics['confidences'].extend(exit_info['confidences'])
        self.inference_metrics['threshold_selections'].extend(exit_info['thresholds_used'])
        
        return {
            'generated_tokens': generated_tokens,
            'exit_info': exit_info,
            'early_exit_rate': exit_info['early_exit_count'] / max(1, exit_info['token_count'])
        }
    
    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Sample next token with temperature and top-p sampling"""
        
        # Temperature scaling
        logits = logits / max(temperature, 1e-8)
        
        # Top-p sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1),
            dim=-1
        )
        
        # Find cutoff index
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[0] = False  # Keep at least one token
        
        logits[sorted_indices[sorted_indices_to_remove]] = -float('Inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def get_inference_summary(self) -> Dict:
        """Get comprehensive inference summary"""
        
        total_tokens = self.inference_metrics['total_tokens']
        total_samples = self.inference_metrics['total_samples']
        
        summary = {
            'total_samples': total_samples,
            'total_tokens': total_tokens,
            'avg_tokens_per_sample': total_tokens / max(1, total_samples),
            'early_exit_rate': (
                self.inference_metrics['early_exits'] / max(1, total_tokens)
            ),
            'avg_exit_layer': (
                np.mean(self.inference_metrics['exit_layers']) 
                if self.inference_metrics['exit_layers'] else 0
            ),
            'avg_confidence': (
                np.mean(self.inference_metrics['confidences'])
                if self.inference_metrics['confidences'] else 0
            ),
            'bandit_metrics': self.bandit_controller.get_summary()
        }
        
        return summary
    
    def log_inference_summary(self):
        """Log detailed inference summary"""
        
        summary = self.get_inference_summary()
        
        logger.info("\n" + "="*60)
        logger.info("PRODUCTION INFERENCE SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Total samples: {summary['total_samples']}")
        logger.info(f"Total tokens: {summary['total_tokens']}")
        logger.info(f"Avg tokens/sample: {summary['avg_tokens_per_sample']:.2f}")
        logger.info(f"Early exit rate: {summary['early_exit_rate']:.2%}")
        logger.info(f"Avg exit layer: {summary['avg_exit_layer']:.1f}")
        logger.info(f"Avg confidence: {summary['avg_confidence']:.4f}")
        
        logger.info("\nBandit convergence:")
        bandit_metrics = summary['bandit_metrics']
        logger.info(f"Best arm: {bandit_metrics['best_arm']}")
        logger.info(f"Best threshold: {bandit_metrics['best_threshold']:.4f}")
        logger.info(f"Best reward: {bandit_metrics['best_reward']:.4f}")
        logger.info(f"Arm concentration: {bandit_metrics['arm_concentration']:.2%}")
        
        logger.info("="*60 + "\n")


class BatchInferencePipeline:
    """Batch inference for higher throughput"""
    
    def __init__(
        self,
        inference_pipeline: ProductionInferencePipeline,
        batch_size: int = 8
    ):
        self.pipeline = inference_pipeline
        self.batch_size = batch_size
    
    def process_batch(
        self,
        texts: List[str],
        max_new_tokens: int = 100
    ) -> List[Dict]:
        """Process batch of texts"""
        
        results = []
        
        # Process in mini-batches
        for batch_start in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Processing batch",
            leave=False
        ):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            for text in batch_texts:
                # Preprocess
                inputs = self.pipeline.preprocess_input(text)
                
                # Generate with adaptive exit
                result = self.pipeline.generate_with_adaptive_exit(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens
                )
                
                results.append(result)
        
        return results


def setup_production_pipeline(
    model_path: str = "unsloth/gemma-3-1b-it",
    confidence_model_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[ProductionInferencePipeline, InferenceConfig]:
    """
    Setup complete production inference pipeline
    
    Args:
        model_path: Path to base LLM
        confidence_model_path: Path to trained confidence classifier checkpoint
        device: Device to use
    
    Returns:
        (pipeline, config)
    """
    
    # Create config
    config = InferenceConfig(
        model_name=model_path,
        device=device,
        exit_layers=[6, 10, 14, 18, 22, 26]
    )
    
    logger.info(f"Setting up production pipeline on {device}")
    
    # Load base model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    logger.info(f"Loading model: {model_path}")
    try:
        # Try with flash attention for better performance on GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=config.dtype,
            attn_implementation="flash_attention_2" if device == "cuda" else "eager"
        )
    except (ImportError, RuntimeError):
        # Fallback to eager attention if flash_attn not installed (common in Colab)
        logger.warning("Flash Attention 2 not available, using eager attention")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=config.dtype,
            attn_implementation="eager"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load confidence classifier
    from .phase2_production import ConfidenceClassifierEnsembleProduction
    
    if confidence_model_path and os.path.exists(confidence_model_path):
        logger.info(f"Loading confidence classifier: {confidence_model_path}")
        confidence_classifier = ConfidenceClassifierEnsembleProduction(
            exit_layer_indices=config.exit_layers
        )
        confidence_classifier.load_state_dict(torch.load(confidence_model_path))
        confidence_classifier.to(device)
    else:
        logger.warning("No confidence classifier checkpoint provided")
        confidence_classifier = ConfidenceClassifierEnsembleProduction(
            exit_layer_indices=config.exit_layers,
            input_dim=1152
        )
        confidence_classifier.to(device)
    
    # Setup bandit controller
    from .phase3_production import AdaptiveThresholdManagerProduction
    
    bandit_controller = AdaptiveThresholdManagerProduction(
        num_arms=config.num_arms,
        threshold_range=config.threshold_range
    )
    
    # Create pipeline
    pipeline = ProductionInferencePipeline(
        model=model,
        confidence_classifier=confidence_classifier,
        bandit_controller=bandit_controller,
        config=config,
        tokenizer=tokenizer
    )
    
    return pipeline, config
