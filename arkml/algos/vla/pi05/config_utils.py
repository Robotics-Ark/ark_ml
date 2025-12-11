import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from omegaconf import OmegaConf


def get_pi05_config() -> Dict[str, Any]:
    """
    Configuration utilities for Pi0.5.

    Returns:
        Configuration dictionary with Pi0.5 specific settings
    """
    # Pi0.5 specific configuration
    config = {
        # Multi-stage training parameters
        'training_stage': 'pretrain',  # 'pretrain' or 'posttrain'
        'pretrain_steps': 280000,
        'posttrain_steps': 80000,
        'integration_steps': 10,  # For flow matching integration
        'flow_alpha': 10.0,  # Weight for flow matching loss

        # Model architecture parameters
        'backbone_type': 'siglip_gemma',  # Vision-language backbone
        'use_fast_tokens': True,  # Whether to use FAST tokenization
        'use_flow_matching': True,  # Whether to use flow matching
        'num_bins': 1000,  # For FAST tokenizer
        'min_action_val': -1.0,
        'max_action_val': 1.0,
    }
    return config


def update_config_for_training_stage(config: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """
    Update configuration based on training stage.

    Args:
        config: Base configuration
        stage: 'pretrain' or 'posttrain'

    Returns:
        Updated configuration for the specific stage
    """
    updated_config = config.copy()
    updated_config['training_stage'] = stage

    if stage == 'pretrain':
        # Pretraining focuses on CE(text) + CE(FAST tokens)
        updated_config['loss_weights'] = {
            'text_ce': 1.0,
            'fast_ce': 1.0,
            'flow_matching': 0.0,
        }
    elif stage == 'posttrain':
        # Post-training focuses on CE(subtask) + alpha * flow_matching_loss
        updated_config['loss_weights'] = {
            'subtask_ce': 1.0,
            'flow_matching': config.get('flow_alpha', 10.0),
        }

    return updated_config