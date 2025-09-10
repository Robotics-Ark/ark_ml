import torch
from arkml.arkml.core.algorithm import Evaluator
from models import masked_l1, kl_loss

class ACTransformerEvaluator(Evaluator):
    def __init__(self, model, dataloader, device="cpu"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.model.eval()

    def evaluate(self, dataloader):
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in dataloader:
                state = batch["state"].to(self.device)
                image = batch["image"].to(self.device)
                target = batch["action_chunk"].to(self.device)
                mask = batch["action_mask"].to(self.device)

                pred, mu, logvar = self.model(image, state, target, mask)
                reconstruction_loss = masked_l1(pred, target, mask)
                kl= kl_loss(mu, logvar).mean()
                loss = reconstruction_loss + kl
                total_loss += loss.item()
                n += 1

        return {"CVAE Loss": total_loss / max(1,n)}
