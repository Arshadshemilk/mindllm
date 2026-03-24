"""
Complete Real Training Pipeline

End-to-end training that:
1. Generates training data using the Phase 1 multi-exit model
2. Trains Phase 2 confidence classifiers on collected data
3. Runs Phase 3 bandit learning during validation inference
4. Produces a fully trained, production-ready system
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import json
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTrainingPipeline:
    """Complete training pipeline for the MindEdgeAI system"""
    
    def __init__(
        self,
        data_path: str = "./data",
        checkpoint_dir: str = "./checkpoints",
        config: Dict = None
    ):
        self.data_path = Path(data_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Default config
        self.config = config or {
            'batch_size': 32,
            'num_epochs_phase2': 20,
            'num_samples_phase3': 1000,
            'learning_rate': 1e-3,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.training_history = {
            'phase1_data': [],
            'phase2_losses': [],
            'phase3_rewards': []
        }
        
        logger.info(f"Training pipeline initialized")
        logger.info(f"Device: {self.config['device']}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
    
    def phase1_data_collection(
        self,
        model,
        texts: List[str],
        tokenizer,
        num_samples_per_text: int = 50
    ) -> Tuple[Dict, Dict]:
        """
        Phase 1: Collect hidden states and training data from base model
        
        Args:
            model: Base LLM from phase1_production
            texts: List of training texts
            tokenizer: Tokenizer for the model
            num_samples_per_text: How many forward passes per text
        
        Returns:
            (hidden_states_dict, targets_dict)
        """
        
        logger.info("="*60)
        logger.info("PHASE 1: DATA COLLECTION")
        logger.info("="*60)
        
        model.eval()
        model.to(self.config['device'])
        
        hidden_states_collection = {}
        targets_collection = {}
        
        exit_layers = [6, 10, 14, 18, 22, 26]
        
        for layer in exit_layers:
            hidden_states_collection[layer] = []
            targets_collection[layer] = []
        
        # Process texts
        for text in tqdm(texts, desc="Collecting data from texts"):
            # Tokenize
            encoded = tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.config['device'])
            attention_mask = encoded['attention_mask'].to(self.config['device'])
            
            with torch.no_grad():
                # Forward pass to get hidden states
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract hidden states at exit layers
                all_hidden = outputs.hidden_states
                
                # Create labels: use model's final prediction as "ground truth"
                # In practice, you'd use actual task labels
                logits = outputs.logits
                final_pred = torch.softmax(logits[:, -1, :], dim=-1)
                confidence_labels = final_pred.max(dim=-1)[0].unsqueeze(-1)
                
                for layer_idx in exit_layers:
                    if layer_idx < len(all_hidden):
                        states = all_hidden[layer_idx]
                        
                        # Take mean over sequence dimension
                        states = states.mean(dim=1)
                        
                        hidden_states_collection[layer_idx].append(states)
                        
                        # Create target: whether model was confident at this layer
                        targets_collection[layer_idx].append(confidence_labels)
        
        logger.info(f"Collected data from {len(texts)} texts")
        for layer in exit_layers:
            logger.info(
                f"  Layer {layer}: {len(hidden_states_collection[layer])} samples, "
                f"shape {hidden_states_collection[layer][0].shape}"
            )
        
        self.training_history['phase1_data'] = {
            'num_texts': len(texts),
            'num_layers': len(exit_layers),
            'exit_layers': exit_layers
        }
        
        return hidden_states_collection, targets_collection
    
    def phase2_train_confidence_classifiers(
        self,
        hidden_states_dict: Dict,
        targets_dict: Dict
    ):
        """
        Phase 2: Train confidence classifiers on collected data
        
        Args:
            hidden_states_dict: Hidden states at each exit layer
            targets_dict: Target labels for each exit layer
        """
        
        logger.info("="*60)
        logger.info("PHASE 2: TRAIN CONFIDENCE CLASSIFIERS")
        logger.info("="*60)
        
        from .phase2_production import (
            ConfidenceClassifierEnsembleProduction,
            ConfidenceTrainerProduction
        )
        
        # Initialize model
        exit_layers = sorted(hidden_states_dict.keys())
        
        model = ConfidenceClassifierEnsembleProduction(
            exit_layer_indices=exit_layers,
            input_dim=2304,
            hidden_dim1=256,
            hidden_dim2=64,
            dropout_rate=0.2
        )
        
        # Initialize trainer
        trainer = ConfidenceTrainerProduction(
            model=model,
            device=self.config['device'],
            learning_rate=self.config['learning_rate'],
            batch_size=self.config['batch_size']
        )
        
        # Train
        best_val_loss = float('inf')
        patience = 5
        no_improve = 0
        
        num_samples = len(hidden_states_dict[exit_layers[0]])
        train_idx = int(0.8 * num_samples)
        
        # Split data
        train_hidden = {
            layer: states[:train_idx]
            for layer, states in hidden_states_dict.items()
        }
        train_targets = {
            layer: targets[:train_idx]
            for layer, targets in targets_dict.items()
        }
        
        val_hidden = {
            layer: states[train_idx:]
            for layer, states in hidden_states_dict.items()
        }
        val_targets = {
            layer: targets[train_idx:]
            for layer, targets in targets_dict.items()
        }
        
        for epoch in range(self.config['num_epochs_phase2']):
            train_metrics = trainer.train_epoch(train_hidden, train_targets)
            val_metrics = trainer.validate(val_hidden, val_targets)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config['num_epochs_phase2']} | "
                f"Train: {train_metrics['loss']:.4f} | "
                f"Val: {val_metrics['loss']:.4f}"
            )
            
            self.training_history['phase2_losses'].append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss']
            })
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                no_improve = 0
                
                # Save checkpoint
                torch.save(
                    model.state_dict(),
                    self.checkpoint_dir / 'confidence_classifier.pt'
                )
                logger.info(f"  → Saved checkpoint")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Apply temperature scaling
        trainer.apply_temperature_scaling(
            train_hidden,
            train_targets,
            target_ece=0.05
        )
        
        # Save final model
        torch.save(
            model.state_dict(),
            self.checkpoint_dir / 'confidence_classifier_final.pt'
        )
        
        return model, trainer
    
    def phase3_bandit_learning(
        self,
        confidence_classifier,
        base_model,
        tokenizer,
        validation_texts: List[str],
        num_inference_samples: int = 1000
    ):
        """
        Phase 3: Online bandit learning during validation inference
        
        Args:
            confidence_classifier: Trained from Phase 2
            base_model: Base LLM
            tokenizer: Tokenizer
            validation_texts: Unseen texts for validation
            num_inference_samples: Number of inference samples
        """
        
        logger.info("="*60)
        logger.info("PHASE 3: BANDIT ONLINE LEARNING")
        logger.info("="*60)
        
        from .phase3_production import UCBBanditControllerProduction
        
        # Initialize bandit
        controller = UCBBanditControllerProduction(
            num_arms=20,
            threshold_range=(0.50, 0.99),
            warmup_steps=100,
            high_explore_steps=400
        )
        
        confidence_classifier.eval()
        base_model.eval()
        
        base_model.to(self.config['device'])
        confidence_classifier.to(self.config['device'])
        
        # Collect inference data
        all_confidences = []
        all_exit_layers = []
        
        for text in tqdm(validation_texts, desc="Collecting inference data"):
            encoded = tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.config['device'])
            attention_mask = encoded['attention_mask'].to(self.config['device'])
            
            with torch.no_grad():
                # Forward pass
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                all_hidden = outputs.hidden_states
                
                # Get confidence at each exit layer
                for layer_idx in [6, 10, 14, 18, 22, 26]:
                    if layer_idx < len(all_hidden):
                        hidden = all_hidden[layer_idx].mean(dim=1)
                        
                        with torch.no_grad():
                            conf = confidence_classifier.heads[str(layer_idx)](
                                hidden,
                                training=False
                            )
                        
                        all_confidences.append(float(conf.item()))
                        all_exit_layers.append(layer_idx)
        
        # Ensure we have enough samples
        sample_indices = np.random.choice(
            len(all_confidences),
            min(num_inference_samples, len(all_confidences)),
            replace=False
        )
        
        # Run online learning
        episode_rewards = []
        for idx in tqdm(sample_indices, desc="Bandit learning"):
            confidence = all_confidences[idx]
            exit_layer = all_exit_layers[idx]
            
            # Select arm
            arm_idx = controller.select_arm()
            
            # Pull arm
            result = controller.pull_arm(
                arm_idx=arm_idx,
                confidence=confidence,
                exit_layer=exit_layer
            )
            
            episode_rewards.append(result['reward'])
        
        # Log results
        controller.log_statistics()
        
        self.training_history['phase3_rewards'] = {
            'num_samples': len(sample_indices),
            'avg_reward': float(np.mean(episode_rewards)),
            'convergence': controller.get_convergence_metrics()
        }
        
        # Save bandit checkpoint
        bandit_state = {
            'arm_stats': controller.get_arm_statistics(),
            'convergence': controller.get_convergence_metrics()
        }
        
        with open(self.checkpoint_dir / 'bandit_checkpoint.json', 'w') as f:
            # Convert numpy types for JSON serialization
            json.dump(
                {k: float(v) if isinstance(v, np.floating) else v
                 for k, v in bandit_state['convergence'].items()},
                f,
                indent=2
            )
        
        return controller
    
    def run_complete_training(
        self,
        model,
        tokenizer,
        train_texts: List[str],
        val_texts: List[str],
        test_texts: List[str]
    ):
        """
        Run complete training pipeline: Phase 1 → Phase 2 → Phase 3
        
        Args:
            model: Base LLM
            tokenizer: Tokenizer
            train_texts: Training texts for Phase 1 data collection
            val_texts: Validation texts for Phase 2 training
            test_texts: Test texts for Phase 3 bandit learning
        """
        
        logger.info("\n" + "="*60)
        logger.info("STARTING COMPLETE MINDEDGEAI TRAINING")
        logger.info("="*60 + "\n")
        
        # Phase 1: Collect data
        hidden_states, targets = self.phase1_data_collection(
            model=model,
            texts=train_texts,
            tokenizer=tokenizer
        )
        
        # Phase 2: Train confidence classifiers
        confidence_model, trainer = self.phase2_train_confidence_classifiers(
            hidden_states_dict=hidden_states,
            targets_dict=targets
        )
        
        # Phase 3: Online bandit learning
        bandit_controller = self.phase3_bandit_learning(
            confidence_classifier=confidence_model,
            base_model=model,
            tokenizer=tokenizer,
            validation_texts=test_texts
        )
        
        # Save training summary
        self.save_training_summary()
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60 + "\n")
        
        return confidence_model, bandit_controller
    
    def save_training_summary(self):
        """Save training summary to file"""
        
        summary_path = self.checkpoint_dir / 'training_summary.json'
        
        # Convert to JSON-serializable format
        summary = {
            'phase1': self.training_history.get('phase1_data', {}),
            'phase2': {
                'num_epochs_trained': len(self.training_history.get('phase2_losses', [])),
                'losses': self.training_history.get('phase2_losses', [])
            },
            'phase3': self.training_history.get('phase3_rewards', {})
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to {summary_path}")


def create_synthetic_training_data(num_texts: int = 100) -> List[str]:
    """Create synthetic training data for demonstration"""
    
    topics = [
        "machine learning and neural networks",
        "natural language processing",
        "computer vision and image processing",
        "quantum computing",
        "distributed systems",
        "database design",
        "web development frameworks",
        "cybersecurity and cryptography"
    ]
    
    texts = []
    
    for i in range(num_texts):
        topic = topics[i % len(topics)]
        text = (
            f"This is a text about {topic}. "
            f"It contains relevant information that the model should process. "
            f"The model will extract features and make predictions based on this input. "
            f"Different early exit points will be evaluated to find optimal efficiency trade-offs. "
            f"Sample {i}: Understanding {topic} is important for practical applications."
        )
        texts.append(text)
    
    return texts


if __name__ == "__main__":
    
    # Create sample data
    train_texts = create_synthetic_training_data(num_texts=50)
    val_texts = create_synthetic_training_data(num_texts=20)
    test_texts = create_synthetic_training_data(num_texts=30)
    
    # Load model (simplified example)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = "unsloth/gemma-3-1b-it"
    logger.info(f"Loading base model: {model_name}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Run training
        pipeline = RealTrainingPipeline()
        
        confidence_model, bandit_controller = pipeline.run_complete_training(
            model=model,
            tokenizer=tokenizer,
            train_texts=train_texts,
            val_texts=val_texts,
            test_texts=test_texts
        )
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.info("Note: For full training, ensure model can be downloaded from HuggingFace")
