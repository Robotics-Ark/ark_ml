import torch

from arkml.core.algorithm import Evaluator


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked L1 loss.

    Computes the mean absolute error between `pred` and `target`,
    considering only positions where `mask=1`.

    Parameters
    ----------
    pred : torch.Tensor, shape (B, K, A)
        Predictions.
    target : torch.Tensor, shape (B, K, A)
        Ground truth.
    mask : torch.Tensor, shape (B, K)
        Binary mask (1 = valid, 0 = ignore).

    Returns
    -------
    torch.Tensor
        Scalar masked L1 loss.
    """
    diff = (pred - target).abs()  # (B,K,A)
    m = mask.unsqueeze(-1)
    num = (diff * m).sum()
    den = (m.sum() * pred.size(-1)).clamp_min(1.0)
    return num / den

def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence loss for Gaussian distributions.

    Computes KL(q||p) where q = N(mu, exp(logvar)) and p = N(0, I).

    Parameters
    ----------
    mu : torch.Tensor, shape (B, D)
        Mean of latent distribution.
    logvar : torch.Tensor, shape (B, D)
        Log-variance of latent distribution.

    Returns
    -------
    torch.Tensor
        KL divergence per sample, shape (B,).
    """
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)  # (B,)


class ACTransformerEvaluator(Evaluator):
    def __init__(self, model, dataloader, device="cpu"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.model.eval()

    def evaluate(self):
        """Run evaluation loop.

        Returns:
        dict: Dictionary with key ``"CVAE Loss"`` containing mean loss.
        """
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.dataloader:
                state = batch["state"].to(self.device)
                image = batch["image"].to(self.device)
                target = batch["action_chunk"].to(self.device)
                mask = batch["action_mask"].to(self.device)

                pred, mu, logvar = self.model(image, state, target, mask)
                reconstruction_loss = masked_l1(pred, target, mask)
                kl = kl_loss(mu, logvar).mean()
                loss = reconstruction_loss + kl
                total_loss += loss.item()
                n += 1

        return {"CVAE Loss": total_loss / max(1, n)}
