import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from arkml.algos.vla.pi05.dataset import Pi05Dataset


def compute_pi05_stats(
    dataset_path: str,
    *,
    obs_dim: int,
    action_dim: int,
    image_shape: Tuple[int, int, int] = (3, 224, 224),
    max_samples: int = 10000,
    save_path: str = None,
    **dataset_kwargs
) -> Dict[str, Any]:
    """
    Compute statistics for Pi0.5 dataset following LeRobot conventions.

    Args:
        dataset_path: Path to the dataset
        obs_dim: Observation dimension
        action_dim: Action dimension
        image_shape: Shape of input images (C, H, W)
        max_samples: Maximum number of samples to use for statistics
        save_path: Optional path to save computed statistics
        **dataset_kwargs: Additional arguments for dataset initialization

    Returns:
        Dictionary containing computed statistics for normalization
    """
    # Initialize dataset
    dataset = Pi05Dataset(dataset_path, **dataset_kwargs)

    # Limit samples for efficiency
    n_samples = min(len(dataset), max_samples)

    # Initialize accumulators for statistics
    action_sum = torch.zeros(action_dim)
    action_sq_sum = torch.zeros(action_dim)
    action_count = 0

    state_sum = torch.zeros(obs_dim)
    state_sq_sum = torch.zeros(obs_dim)
    state_count = 0

    # Process samples to compute statistics
    for i in range(n_samples):
        sample = dataset[i]

        # Compute action statistics
        if "action" in sample:
            action = sample["action"]
            if torch.is_tensor(action):
                action = action.float()
            else:
                action = torch.tensor(action, dtype=torch.float32)

            action_sum += action
            action_sq_sum += action ** 2
            action_count += 1

        # Compute state statistics
        if "observation.state" in sample:
            state = sample["observation.state"]
            if torch.is_tensor(state):
                state = state.float()
            else:
                state = torch.tensor(state, dtype=torch.float32)

            state_sum += state
            state_sq_sum += state ** 2
            state_count += 1

    # Calculate mean and std for actions
    if action_count > 0:
        action_mean = action_sum / action_count
        action_var = (action_sq_sum / action_count) - (action_mean ** 2)
        action_std = torch.sqrt(torch.clamp(action_var, min=1e-8))
    else:
        action_mean = torch.zeros(action_dim)
        action_std = torch.ones(action_dim)

    # Calculate mean and std for states
    if state_count > 0:
        state_mean = state_sum / state_count
        state_var = (state_sq_sum / state_count) - (state_mean ** 2)
        state_std = torch.sqrt(torch.clamp(state_var, min=1e-8))
    else:
        state_mean = torch.zeros(obs_dim)
        state_std = torch.ones(obs_dim)

    # Create statistics dictionary in LeRobot format
    stats = {
        "observation.state": {
            "mean": state_mean.tolist(),
            "std": state_std.tolist(),
            "min": state_mean.tolist(),  # Placeholder - in real impl, compute actual min/max
            "max": state_mean.tolist()   # Placeholder - in real impl, compute actual min/max
        },
        "observation.images.image": {
            "mean": [0.485, 0.456, 0.406],  # ImageNet normalization values as placeholder
            "std": [0.229, 0.224, 0.225],   # ImageNet normalization values as placeholder
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0]
        },
        "action": {
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
            "min": torch.min(action_mean - 3 * action_std).item(),  # Estimate from mean and std
            "max": torch.max(action_mean + 3 * action_std).item()
        }
    }

    # Save statistics if path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)

    return stats


def load_pi05_stats(stats_path: str) -> Dict[str, Any]:
    """
    Load pre-computed Pi0.5 dataset statistics.

    Args:
        stats_path: Path to the statistics file

    Returns:
        Dictionary containing loaded statistics
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats


def normalize_action(action: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
    """
    Normalize action using computed statistics.

    Args:
        action: Raw action tensor
        stats: Statistics dictionary

    Returns:
        Normalized action tensor
    """
    action_mean = torch.tensor(stats["action"]["mean"], dtype=action.dtype, device=action.device)
    action_std = torch.tensor(stats["action"]["std"], dtype=action.dtype, device=action.device)

    # Clamp normalized values to reasonable range to avoid outliers
    normalized = (action - action_mean) / torch.clamp(action_std, min=1e-8)
    return torch.clamp(normalized, min=-10.0, max=10.0)  # Clamp to reasonable range


def unnormalize_action(normalized_action: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
    """
    Unnormalize action using computed statistics.

    Args:
        normalized_action: Normalized action tensor
        stats: Statistics dictionary

    Returns:
        Unnormalized action tensor
    """
    action_mean = torch.tensor(stats["action"]["mean"], dtype=normalized_action.dtype, device=normalized_action.device)
    action_std = torch.tensor(stats["action"]["std"], dtype=normalized_action.dtype, device=normalized_action.device)

    return normalized_action * action_std + action_mean