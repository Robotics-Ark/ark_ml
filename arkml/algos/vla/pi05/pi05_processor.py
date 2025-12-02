"""
Pi0.5 Processor for handling image and text preprocessing
This is a minimal processor implementation that will be enhanced with CLIP + Qformer later.
"""

from typing import Dict, Any, Union
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

class Pi05Processor:
    """
    Processor for Pi0.5 that handles image resizing/normalization and text tokenization.
    This serves as a minimal implementation to be replaced with full CLIP + Qformer later.
    """
    
    def __init__(
        self, 
        image_size: tuple = (224, 224),
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
        device: str = "cpu"
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.max_length = max_length
        self.device = device
        
        # Create image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Initialize tokenizer - for now, use our own dummy tokenizer instead of HuggingFace
        # This ensures compatibility with the model's embedding layer
        self.tokenizer = None  # Disable HuggingFace tokenizer
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
        }
        for i, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:"):
            self.vocab[char] = len(self.vocab)

    def preprocess_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image or tensor
            
        Returns:
            Normalized image tensor
        """
        if isinstance(image, Image.Image):
            # Apply transforms to PIL image
            image_tensor = self.image_transform(image)
        elif torch.is_tensor(image):
            # Apply normalization to existing tensor
            image_tensor = image
            if image_tensor.dim() == 3:  # (C, H, W) - add batch dim
                image_tensor = image_tensor.unsqueeze(0)
            # Normalize if not already normalized (assuming input is [0,1] or [0,255])
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0  # Scale from [0,255] to [0,1]
            # Apply normalize transform manually for consistency
            mean = torch.tensor(self.mean, device=image_tensor.device).view(3, 1, 1)
            std = torch.tensor(self.std, device=image_tensor.device).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        return image_tensor.to(self.device)

    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        Preprocess text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            Token IDs tensor
        """
        if self.tokenizer is not None:
            # Use HuggingFace tokenizer
            tokens = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return tokens['input_ids'].squeeze(0).to(self.device)
        else:
            # Fallback to simple character-level tokenization
            tokens = []
            for char in text.lower():
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                else:
                    tokens.append(self.vocab["<unk>"])

            # Add padding/truncation
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens.extend([self.vocab["<pad>"]] * (self.max_length - len(tokens)))

            # Ensure all tokens are within embedding range (0 to embedding vocab size - 1)
            # The model's text embedding has vocab_size 128 in our implementation
            tokens = [min(token, 127) for token in tokens]  # Clamp to max vocab index (128-1)

            return torch.tensor(tokens, dtype=torch.long, device=self.device)

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess observation dict containing image and optionally text.
        
        Args:
            obs: Observation dict with 'image' key and optionally 'language' or 'instruction'
            
        Returns:
            Preprocessed observation dict
        """
        processed = {}
        
        # Process image
        if 'image' in obs:
            processed['image'] = self.preprocess_image(obs['image'])
        elif 'observation.images.image' in obs:
            processed['image'] = self.preprocess_image(obs['observation.images.image'])
        else:
            raise ValueError("Image key not found in observation")
            
        # Process text
        text_key = None
        if 'language' in obs:
            text_key = 'language'
        elif 'instruction' in obs:
            text_key = 'instruction'
        elif 'task_description' in obs:
            text_key = 'task_description'
            
        if text_key:
            processed['instruction'] = self.preprocess_text(obs[text_key])
        else:
            # Create dummy text tokens if no text provided
            dummy_text = "default instruction"
            processed['instruction'] = self.preprocess_text(dummy_text)
            
        return processed