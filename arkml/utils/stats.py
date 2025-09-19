from typing import Any

import numpy as np


def _init_state(shape: tuple[int, Any], dtype: np.dtype = np.float64) -> dict[str, Any]:
    """
    Initialize a state dictionary for accumulating running statistics.
    Args:
        shape: The shape of the feature dimension(s) for which statistics will
        be tracked. For example, (d,) for a d-dimensional vector.
        dtype:  The NumPy data type used for the arrays

    Returns:

    """
    return {
        "count": 0,
        "sum": np.zeros(shape, dtype=dtype),
        "sumsq": np.zeros(shape, dtype=dtype),
        "min": np.full(shape, np.inf, dtype=dtype),
        "max": np.full(shape, -np.inf, dtype=dtype),
    }


def _accumulate_moments(x: np.ndarray, state: dict[str, Any]) -> None:
    """
    Update running statistics (count, sum, sum of squares, min, max) in-place for a given batch of data.
    Args:
        x: Input data of shape (n, d) or (d,), where n is the number of samples and d is the feature dimension.
        state: A dictionary containing running statistics.

    Returns:
        None

    """
    x = np.asarray(x)
    n = x.shape[0]
    state["count"] += n
    state["sum"] += x.sum(axis=0)
    state["sumsq"] += np.square(x).sum(axis=0)
    state["min"] = np.minimum(state["min"], x.min(axis=0))
    state["max"] = np.maximum(state["max"], x.max(axis=0))


def _finalize_stats(state: dict[str, Any]) -> dict[str, np.ndarray]:
    """
     Compute final dataset statistics (mean, std, min, max, count) from accumulated running moments.
    Args:
        state: A dictionary containing accumulated statistics.

    Returns:
         A dictionary with finalized statistics:
            - "min": np.ndarray, element-wise minimum
            - "max": np.ndarray, element-wise maximum
            - "mean": np.ndarray, element-wise mean
            - "std": np.ndarray, element-wise standard deviation
            - "count": np.ndarray of shape (1,), total number of samples

    """

    count = max(1, int(state["count"]))
    mean = state["sum"] / count
    var = np.maximum(0.0, state["sumsq"] / count - np.square(mean))
    std = np.sqrt(var)
    return {
        "min": state["min"],
        "max": state["max"],
        "mean": mean,
        "std": std,
        "count": np.array([count], dtype=np.int64),
    }


def estimate_num_samples(
    dataset_len: int,
    min_num_samples: int = 100,
    max_num_samples: int = 10_000,
    power: float = 0.75,
) -> int:
    """
    Estimate a reasonable number of samples to draw from a dataset.
    The estimate is based on a power-law scaling of the dataset size,
    with lower and upper bounds applied to avoid extreme values.
    Args:
        dataset_len: Total number of items in the dataset.
        min_num_samples:  Minimum number of samples to return. If the dataset is smaller
        than this value, the estimate will not exceed `dataset_len`.
        max_num_samples: Maximum number of samples to return.
        power: Exponent used for power-law scaling of the dataset size.
        For example, `dataset_len**0.75`. Default is 0.75.

    Returns:
         Estimated number of samples,

    """
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    """
    Generate evenly spaced sample indices for a dataset.
    Args:
        data_len: Total number of items in the dataset.

    Returns:
        A list of integer indices in the range [0, data_len - 1].

    """
    num_samples = estimate_num_samples(data_len)
    if data_len <= 1:
        return [0]
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()
