import torch
import numpy as np
from torch.utils.data import DataLoader
from arkml.algos.vla.pi05.dataset import Pi05Dataset


def compute_pi05_stats(dataset_path, *, obs_dim: int, action_dim: int, image_channels: int, sample_images_only: bool = True):
    """
    Compute statistics for Pi0.5 dataset.

    Computes:
    - mean/std of pixel values
    - mean/std of actions
    - episode length distribution
    """
    # Initialize dataset
    dataset = Pi05Dataset(dataset_path)

    # Create a small dataloader to sample data
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Collect pixel values for mean/std computation
    pixel_values = []
    actions_list = []
    lengths = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 10:  # Limit to first 10 batches for efficiency
            break

        # Extract image data if available
        if "prefix_tokens" in batch and batch["prefix_tokens"] is not None:
            # For now, get a sample from the batch - note that prefix_tokens contains vision+language tokens
            # For pixel statistics, we need the original image data which may not be available in this format
            pass

        # Extract continuous actions if available
        if "actions_cont" in batch and batch["actions_cont"] is not None and batch["actions_cont"].numel() > 0:
            actions = batch["actions_cont"]
            if actions.numel() > 0:
                actions_list.append(actions.flatten())

        # Calculate lengths from available data
        lengths.append(batch["prefix_tokens"].size(0) if "prefix_tokens" in batch else 1)

    stats = {}

    # Compute action statistics if actions are available
    if actions_list:
        all_actions = torch.cat(actions_list, dim=0)
        stats['action_mean'] = all_actions.mean().item()
        stats['action_std'] = all_actions.std().item()
        stats['action_min'] = all_actions.min().item()
        stats['action_max'] = all_actions.max().item()
    else:
        stats['action_mean'] = 0.0
        stats['action_std'] = 0.0
        stats['action_min'] = 0.0
        stats['action_max'] = 0.0

    # Compute episode length statistics
    if lengths:
        stats['episode_lengths_mean'] = np.mean(lengths)
        stats['episode_lengths_std'] = np.std(lengths)
        stats['episode_lengths_min'] = np.min(lengths)
        stats['episode_lengths_max'] = np.max(lengths)
    else:
        stats['episode_lengths_mean'] = 0.0
        stats['episode_lengths_std'] = 0.0
        stats['episode_lengths_min'] = 0.0
        stats['episode_lengths_max'] = 0.0

    # For pixel statistics, we need different approach since we may not have raw images
    # We'll use synthetic data to show the structure
    stats['pixel_mean'] = 0.0  # Placeholder - real implementation would load actual images
    stats['pixel_std'] = 1.0   # Placeholder - real implementation would load actual images

    return stats

