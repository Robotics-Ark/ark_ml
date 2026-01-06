import ast
import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torchvision import transforms


class ConfigPath:
    """
    A utility class to handle configuration file paths and reading.
    """
    def __init__(self, path: str):
        self.path = Path(path)

    def read_yaml(self) -> dict:
        """
        Read and parse a YAML configuration file.

        Returns:
            The parsed configuration as a dictionary.
        """
        if self.path.exists():
            with open(self.path, "r") as f:
                cfg_dict = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file could not be found {self.path}")

        return cfg_dict


def _normalise_shape(shape_dim: str) -> tuple:
    """
    Parse a shape string into a normalized tuple of dimensions.
    Args:
        shape_dim: A string representation of a shape, e.g. "(3, 224, 224)" or "[64, 128]".

    Returns:
        A tuple of integers if the string could be parsed into a list or tuple.

    """
    try:
        parsed = ast.literal_eval(shape_dim)
        if isinstance(parsed, (list, tuple)):
            return tuple(parsed)
        else:
            raise ValueError(f"shape {shape_dim} failed to convert to a list/tuple")
    except (ValueError, TypeError):
        raise ValueError(f"shape {shape_dim} failed to convert to a list/tuple")


def _resolve_channel_types(mapping: dict[str, Any]) -> dict[str, type]:
    """
    Resolve a mapping of channel names to Ark types.
    Accepts either already-imported classes or string names present in the
    ``arktypes`` package. Returns a mapping of channel name to type.
    Args:
        A dictionary mapping channel names to resolved Ark type objects.

    Returns:

    """
    if not mapping:
        return {}
    resolved: dict[str, type] = {}
    arktypes_mod = importlib.import_module("arktypes")
    for ch_name, t in mapping.items():
        if isinstance(t, str):
            resolved[ch_name] = getattr(arktypes_mod, t)
        else:
            resolved[ch_name] = t
    return resolved


def print_trainable_summary(model: nn.Module) -> None:
    """
    Print a detailed summary of a model’s parameters.
    Args:
        model: A PyTorch model.

    Returns:
        None

    """
    total_params = 0
    trainable_params = 0
    is_debug = os.getenv("DEBUG", False) in ["True", "true", "1"]

    print("\n=== Trainable Parameters Summary ===")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            if is_debug:
                print(
                    f"[TRAINABLE] {name:50} | shape={tuple(param.shape)} | params={num_params:,}"
                )
        else:
            if is_debug:
                print(
                    f"[frozen   ] {name:50} | shape={tuple(param.shape)} | params={num_params:,}"
                )

    print("\n--- Totals ---")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
    print("==============================\n")


def _image_to_tensor(image_value: Any, transform: transforms = None) -> torch.Tensor:
    array = np.asarray(image_value)
    if array.dtype != np.uint8:
        array_float = array.astype(np.float32)
        if array_float.max() <= 1.0:
            array_uint8 = np.clip(array_float * 255.0, 0, 255).astype(np.uint8)
        else:
            array_uint8 = np.clip(array_float, 0, 255).astype(np.uint8)
    else:
        array_uint8 = array

    transform = transform or transforms.ToTensor()

    image = Image.fromarray(array_uint8)
    return transform(image)
