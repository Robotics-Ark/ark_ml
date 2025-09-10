import torch
from ark_ml.arkml.algos.ACTransformer.models import ACT
from ark_ml.arkml.nodes.policy_node import PolicyNode

class ActPolicyNode(PolicyNode):
    def __init__(self, model_cfg, device="cuda"):

        policy = ACT(
            joint_dim=model_cfg.joint_dim,
            action_dim=model_cfg.action_dim,
            z_dim=model_cfg.z_dim,
            d_model=model_cfg.d_model,
            ffn_dim=model_cfg.ffn_dim,
            nhead=model_cfg.nhead,
            enc_layers=model_cfg.enc_layers,
            dec_layers=model_cfg.dec_layers,
            dropout=model_cfg.dropout,
            max_len=model_cfg.max_len,
            img_channels=model_cfg.img_channels,
        )
        super().__init__(policy=policy, device=device)

    def predict(self, obs):
        with torch.no_grad():
            action = self.policy.forward(obs)
        return action.detach().cpu().numpy()