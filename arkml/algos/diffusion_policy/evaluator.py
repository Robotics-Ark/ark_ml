import torch
import torch.nn.functional as F

from ark_ml.arkml.core.algorithm import Evaluator


class DiffusionEvaluator(Evaluator):
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
                obs = batch['obs'].to(self.device)
                action = batch['action'].to(self.device)
                noise = torch.randn_like(action)
                timesteps = torch.randint(0, 100, (action.shape[0],), device=self.device).long()
                noisy_action = action + noise
                pred_noise = self.model(noisy_action, timesteps, obs.flatten(start_dim=1))
                loss = F.mse_loss(pred_noise, noise, reduction='sum')
                total_loss += loss.item()
                n += action.shape[0]
        return {'mse_loss': total_loss / n}
