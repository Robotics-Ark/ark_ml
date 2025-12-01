import json
import os
import random
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from arkml.algos.vla.tokenizers.fast import FASTTokenizer
from PIL import Image
import torchvision.transforms as transforms


class Pi05Dataset(Dataset):
    """
    Dataset class for Pi0.5 supporting multiple modalities.

    Supports sampling from these modalities:
    - web_caption
    - qa
    - bounding_boxes
    - hl_subtask
    - fast_robot_actions
    - continuous_robot_actions
    """

    def __init__(
        self,
        dataset_path: str,
        config_path: str = "arkml/configs/data/pi05_dataset.yaml",
        transform=None,
        pred_horizon: int = 1,
        tokenizer_vocab_path: str = "",
        num_bins: int = 1000,
        min_val: float = -1.0,
        max_val: float = 1.0
    ):
        self.dataset_path = dataset_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.pred_horizon = pred_horizon

        # Load the configuration
        self.config = OmegaConf.load(config_path)

        # Initialize mixture sampling based on config
        self.mixture_config = self.config.dataset.mixture
        self.primary_dataset = self.mixture_config.primary_dataset
        self.secondary_datasets = self.mixture_config.secondary_datasets
        self.weights = self.mixture_config.weights

        # Calculate sampling weights
        self.primary_weight = self.weights.primary
        self.secondary_weight = self.weights.secondary if 'secondary' in self.weights else 0.3
        total_secondary_weight = self.secondary_weight / len(self.secondary_datasets) if self.secondary_datasets else 0

        # Calculate cumulative weights for sampling
        self.dataset_weights = [self.primary_weight]
        for i in range(len(self.secondary_datasets)):
            self.dataset_weights.append(self.dataset_weights[-1] + total_secondary_weight)

        # FAST tokenizer for action conversion (for pretrain stage)
        self.fast_tokenizer = FASTTokenizer(
            vocab_path=tokenizer_vocab_path,
            num_bins=num_bins,
            min_val=min_val,
            max_val=max_val
        )

        # Define supported modalities
        self.modalities = [
            "web_caption",
            "qa",
            "bounding_boxes",
            "hl_subtask",
            "fast_robot_actions",
            "continuous_robot_actions"
        ]

        # Placeholder for dataset loading logic
        # In a real implementation, this would load trajectories from the dataset_path
        # For now we'll create placeholders for the different modalities
        self.dataset_samples = self._load_samples()

    def _load_samples(self):
        """
        Load dataset samples from the specified path.
        This is a placeholder - in real implementation this would load actual trajectories.
        """
        # Placeholder implementation - in reality this would load from actual dataset files
        samples = []

        # Simulate a few samples for each modality
        for modality in self.modalities:
            # Create mock samples based on the modality type
            num_samples = 100  # Placeholder - would be actual count in real implementation
            for i in range(num_samples):
                sample = {
                    "modality": modality,
                    "dataset_type": "primary" if i < 70 else "secondary",  # Simulate mixture
                    "index": i
                }

                # Add modality-specific mock data
                if modality in ["web_caption", "qa", "hl_subtask"]:
                    sample["text"] = f"sample text for {modality} {i}"
                elif modality == "bounding_boxes":
                    sample["bbox"] = np.random.rand(4).tolist()  # x, y, w, h
                elif modality in ["fast_robot_actions", "continuous_robot_actions"]:
                    # Sample random continuous actions
                    sample["actions_cont"] = np.random.rand(8).tolist()  # 8-dim action space

                # Mock image path
                sample["image_path"] = f"mock_image_{modality}_{i}.jpg"

                samples.append(sample)

        return samples

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.dataset_samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            dict: Dictionary containing:
                - "prefix_tokens": Vision + language tokens for prefix
                - "target_tokens": Target tokens (actions or text)
                - "modality": The modality type
                - "actions_cont": Continuous action values
        """
        sample = self.dataset_samples[idx]
        modality = sample["modality"]

        # Load image using PIL
        try:
            image = self._load_image(sample["image_path"])
        except:
            # Fallback to random tensor if image loading fails
            image = torch.rand(3, 224, 224)

        # Convert image to vision tokens (placeholder - leave TODO)
        # TODO: Implement actual image to vision tokens conversion
        vision_tokens = torch.zeros(100)  # Placeholder for vision tokens

        # Convert text to language tokens (placeholder - leave TODO)
        # TODO: Implement actual text to language tokens conversion
        language_tokens = torch.zeros(50)  # Placeholder for language tokens

        # Combine prefix tokens (vision + language)
        prefix_tokens = torch.cat([vision_tokens, language_tokens])

        # Handle target tokens based on modality
        if modality in ["fast_robot_actions", "continuous_robot_actions"]:
            # Convert continuous actions using FAST tokenizer for pretrain stage
            actions_cont = torch.tensor(sample.get("actions_cont", [0.0] * 8), dtype=torch.float32)

            # Use FAST tokenizer to convert continuous actions to tokens (for pretrain stage)
            # For now, just return continuous actions and tokens
            action_tokens_list = self.fast_tokenizer.encode(actions_cont.numpy())
            target_tokens = torch.tensor(action_tokens_list, dtype=torch.long)
        else:
            # For other modalities, target might be text tokens (placeholder)
            target_tokens = torch.zeros(10, dtype=torch.long)  # Placeholder
            actions_cont = torch.zeros(8, dtype=torch.float32)  # Placeholder when not available

        return {
            "prefix_tokens": prefix_tokens,
            "target_tokens": target_tokens,
            "modality": modality,
            "actions_cont": actions_cont if 'actions_cont' in locals() else torch.zeros(8, dtype=torch.float32)
        }

    def _load_image(self, image_path: str):
        """
        Load and preprocess image from path.
        """
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image