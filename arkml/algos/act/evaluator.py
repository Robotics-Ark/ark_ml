import torch

from arkml.core.algorithm import Evaluator


def masked_l1(pred, target, mask):
    diff = (pred - target).abs()  # (B,K,A)
    m = mask.unsqueeze(-1)
    num = (diff * m).sum()
    den = (m.sum() * pred.size(-1)).clamp_min(1.0)
    return num / den


def kl_loss(mu, logvar):
    # KL(q||p), p=N(0,I)
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
