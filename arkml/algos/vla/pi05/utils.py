import torch
import torch.nn.functional as F


def flow_matching_loss(pred, target):
    """
    Compute flow matching loss between predicted and target actions.

    Args:
        pred: Predicted flow vectors or actions
        target: Target flow vectors or actions

    Returns:
        Scalar loss value (MSE loss)
    """
    return F.mse_loss(pred, target)


def euler_integration_step(initial_state, steps: int = 10, step_size: float = 0.1, vector_field_fn=None):
    """
    Perform Euler integration for flow matching.

    Args:
        initial_state: Starting state for integration
        steps: Number of integration steps
        step_size: Size of each integration step
        vector_field_fn: Function that computes the vector field

    Returns:
        Integrated result
    """
    current_state = initial_state.clone()

    for _ in range(steps):
        if vector_field_fn:
            flow_vector = vector_field_fn(current_state)
            current_state = current_state + step_size * flow_vector
        else:
            # Default: identity transformation
            break

    return current_state