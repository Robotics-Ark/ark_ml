import json
import os
from typing import Dict, List, Any, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from arkml.algos.vla.tokenizers.fast import FASTTokenizer


class Pi05Dataset(Dataset):
    """
    Dataset class for Pi0.5 supporting multiple modalities.
    Designed to work with LeRobot-based Pi0.5 policy.

    Supports sampling from these modalities:
    - web_caption
    - qa
    - hl_subtask
    - fast_robot_actions
    - continuous_robot_actions
    """

    def __init__(
        self,
        dataset_path: str,
        obs_horizon: int = 1,
        pred_horizon: int = 1,
        image_keys: List[str] = ["image"],
        state_keys: List[str] = ["state"],
        action_keys: List[str] = ["action"],
        tokenizer_vocab_path: str = "",
        num_bins: int = 1000,
        min_val: float = -1.0,
        max_val: float = 1.0
    ):
        self.dataset_path = dataset_path
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.image_keys = image_keys
        self.state_keys = state_keys
        self.action_keys = action_keys

        # FAST tokenizer for action conversion during pretrain stage
        self.fast_tokenizer = FASTTokenizer(
            vocab_path=tokenizer_vocab_path,
            num_bins=num_bins,
            min_val=min_val,
            max_val=max_val
        )

        # Load and validate dataset
        self._load_dataset()

    def _load_dataset(self):
        """
        Load dataset from the specified path.
        This method should be implemented to load actual trajectories.
        """
        # In a real implementation, this would load LeRobot-compatible datasets
        # For now we'll set up placeholders to demonstrate the structure
        # This would typically interface with LeRobot's dataset loading utilities

        # Placeholder: In real implementation, this would load from LeRobot dataset
        # Example: self.dataset = LeRobotDataset.create_dataset_from_configs(...)
        self.dataset_length = 1000  # Placeholder - actual length from real dataset

        # The dataset should provide trajectories with:
        # - Images: (T, C, H, W)
        # - States: (T, state_dim)
        # - Actions: (T, action_dim)
        # Where T is the trajectory length

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            dict: Dictionary containing:
                - "observation.images.image": Image tensor
                - "observation.state": State vector
                - "action": Action vector
                - "modality": Modality type for multi-stage training
                - "prefix_tokens": For pretrain stage
                - "target_tokens": For pretrain stage
                - "observation.language.tokens": Language token tensor
                - "observation.language.attention_mask": Attention mask tensor
        """
        # In real implementation, load actual trajectory data at index `idx`
        # For demonstration, create mock data that matches LeRobot Pi0.5 expectations

        # Mock image observation
        image = torch.randn(3, 224, 224)  # Image tensor (C, H, W)

        # Mock state observation
        state = torch.randn(9)  # State vector

        # Mock action
        action = torch.randn(8)  # Action vector

        # Randomly assign a modality for multi-stage training
        modalities = ["web_caption", "qa", "hl_subtask", "fast_robot_actions", "continuous_robot_actions"]
        modality_idx = idx % len(modalities)
        modality = modalities[modality_idx]

        # For pretraining stage - convert continuous actions to FAST tokens
        fast_tokens = torch.tensor(
            self.fast_tokenizer.encode(action.numpy()),
            dtype=torch.long
        )

        # For post-training stage - keep continuous actions
        actions_cont = action

        # Mock language tokens - simulate variable length sequences
        # In real implementation, this would come from the actual language data
        language_seq_len = np.random.randint(10, 50)  # Variable length between 10-50
        language_tokens = torch.randint(0, 1000, (language_seq_len,), dtype=torch.long)  # Random tokens
        attention_mask = torch.ones(language_seq_len, dtype=torch.long)  # All tokens are valid

        sample = {
            "observation.images.image": image,
            "observation.state": state,
            "action": action,
            "modality": [modality],  # Using list to match expected format
            "prefix_tokens": torch.zeros(50, dtype=torch.long),  # Placeholder
            "target_tokens": fast_tokens if modality == "fast_robot_actions" else torch.zeros(10, dtype=torch.long),
            "actions_cont": actions_cont,
            "observation.language.tokens": language_tokens,
            "observation.language.attention_mask": attention_mask
        }

        return sample


def create_pi05_dataloader(
    dataset_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a dataloader for Pi0.5 dataset.

    Args:
        dataset_path: Path to the dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        **kwargs: Additional arguments for dataset initialization

    Returns:
        DataLoader configured for Pi0.5
    """
    dataset = Pi05Dataset(dataset_path, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pi05_collate_fn  # Custom collate function if needed
    )


def pi05_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for Pi0.5 dataset.
    Handles batching of different modalities and sequence lengths.
    Specifically handles variable-length language tokens and attention masks.
    """
    if not batch:
        return {}

    # Stack tensors that should be batched
    collated_batch = {}

    # Keys that need to be stacked (fixed size)
    stack_keys = ["observation.images.image", "observation.state", "action", "actions_cont"]

    # Keys that might be single values per batch
    single_keys = ["modality"]

    # Keys that might have different lengths (for tokenization)
    variable_keys = ["prefix_tokens", "target_tokens"]

    # Language-specific keys that need special handling for padding
    language_keys = ["observation.language.tokens", "observation.language.attention_mask"]

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if key in stack_keys:
            # Stack tensors of the same size
            try:
                collated_batch[key] = torch.stack(values, dim=0)
            except RuntimeError:
                # If they have different sizes, pad them (for variable length data)
                max_len = max([v.shape[0] if v.dim() > 0 else 1 for v in values])
                padded_values = []
                for v in values:
                    if v.dim() == 0:  # scalar
                        v = v.unsqueeze(0)
                    if v.shape[0] < max_len:
                        # Pad to max length
                        padding_size = [max_len - v.shape[0]] + list(v.shape[1:])
                        v = torch.cat([v, torch.zeros(*padding_size, dtype=v.dtype)], dim=0)
                    padded_values.append(v)
                collated_batch[key] = torch.stack(padded_values, dim=0)
        elif key in single_keys:
            # For single values like modality, return as is or take first
            collated_batch[key] = values  # Keep as list to preserve individual values
        elif key in variable_keys:
            # Handle variable length sequences (token sequences)
            max_len = max([v.shape[0] if v.dim() > 0 else 1 for v in values])
            padded_values = []
            for v in values:
                if v.dim() == 0:  # scalar
                    v = v.unsqueeze(0)
                if v.shape[0] < max_len:
                    # Pad to max length with padding token (0)
                    padding_size = [max_len - v.shape[0]]
                    v = torch.cat([v, torch.zeros(*padding_size, dtype=v.dtype, device=v.device)], dim=0)
                padded_values.append(v)
            collated_batch[key] = torch.stack(padded_values, dim=0)
        elif key in language_keys:
            # Handle language tokens and attention masks with special padding logic
            # Both tokens and attention_mask should have the same sequence length per item
            max_len = max([v.shape[0] if v.dim() > 0 else 1 for v in values])
            padded_values = []
            for v in values:
                if v.dim() == 0:  # scalar
                    v = v.unsqueeze(0)
                if v.shape[0] < max_len:
                    # Pad to max length - for tokens use 0 (pad token), for attention_mask use 0 (ignore)
                    padding_size = [max_len - v.shape[0]] + list(v.shape[1:])
                    v = torch.cat([v, torch.zeros(*padding_size, dtype=v.dtype, device=v.device)], dim=0)
                padded_values.append(v)
            collated_batch[key] = torch.stack(padded_values, dim=0)
        else:
            # For other keys, stack if possible
            try:
                collated_batch[key] = torch.stack(values, dim=0)
            except RuntimeError:
                # If they can't be stacked, keep as list
                collated_batch[key] = values

    return collated_batch