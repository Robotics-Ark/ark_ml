from torch import nn


def print_trainable_summary(root: nn.Module) -> None:
    """Print a summary of trainable vs total parameters.

    Computes the total number of parameters in a module, the subset marked as
    trainable (i.e., requires_grad=True), and prints a formatted summary with
    counts and the trainable percentage.

    Args:
        root: The root module whose parameters are inspected.

    Returns:
        None
    """
    total = sum(p.numel() for p in root.parameters())
    trainable = sum(p.numel() for p in root.parameters() if p.requires_grad)
    print(
        f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100.0 * trainable / total:.4f}"
    )
