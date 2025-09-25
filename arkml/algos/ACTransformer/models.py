from typing import Any

import torch
import torch.nn as nn
from arkml.core.policy import BasePolicy
from arkml.core.registry import MODELS
from torchvision.models import resnet18, ResNet18_Weights


def sinusoid_1d(length, dim, device, dtype=torch.float32):
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)  # (L,1)
    i = torch.arange(dim, device=device, dtype=dtype).unsqueeze(0)  # (1,D)
    angle_rates = torch.pow(
        torch.tensor(10000.0, device=device, dtype=dtype), -(2 * (i // 2)) / dim
    )
    angles = pos * angle_rates  # (L,D)
    pe = torch.zeros((length, dim), device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe  # (L,D)


def sinusoid_2d(h, w, dim, device, dtype=torch.float32):
    assert dim % 2 == 0
    pe_h = sinusoid_1d(h, dim // 2, device, dtype)[:, None, :].repeat(
        1, w, 1
    )  # (H,W, D/2)
    pe_w = sinusoid_1d(w, dim // 2, device, dtype)[None, :, :].repeat(
        h, 1, 1
    )  # (H,W,D/2)
    pe = torch.cat([pe_h, pe_w], dim=-1)  # (H,W,D)
    return pe.view(h * w, dim)  # (HW,D)


class ResNet18Tokens(nn.Module):
    def __init__(self, d_model=512, freeze_backbone=True, freeze_bn=True, pretrained=True):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )

        self.frozen = bool(freeze_backbone)

        # Freeze the whole backbone (params + keep BN in eval)
        if self.frozen:
            for p in self.stem.parameters():
                p.requires_grad = False
            self.stem.eval()

        # Optionally also lock BN even if not fully frozen
        if freeze_bn:
            self._freeze_bn(self.stem)

        self.proj = nn.Conv2d(512, d_model, kernel_size=1)
        self.d_model = d_model
        self._cached = None  # (h, w, device, dtype, pe)

    def _freeze_bn(self, m):
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
                for p in mod.parameters():
                    p.requires_grad = False

    # Keep BN/backbone in eval even if model.train() is called
    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.stem.eval()
        return self

    def forward(self, x):
        """
        x: (B,3,H,W) -> tokens: (B, HW, d_model)
        """
        if self.frozen:
            with torch.no_grad():
                feat = self.stem(x)                # (B,512,h,w)
        else:
            feat = self.stem(x)

        feat = self.proj(feat)             # (B,d,h,w)
        B, d, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, hw, d)

        key = (h, w, tokens.device, tokens.dtype)
        if (self._cached is None) or (self._cached[:4] != key):
            pe = sinusoid_2d(h, w, d, tokens.device, tokens.dtype)  # (hw,d)
            self._cached = (h, w, tokens.device, tokens.dtype, pe)

        tokens = tokens + self._cached[4].unsqueeze(0)
        return tokens  # (B, hw, d_model)


@MODELS.register("act")
class ACT(BasePolicy):

    def __init__(
        self,
        joint_dim=9,
        action_dim=8,
        z_dim=32,
        d_model=512,
        ffn_dim=3200,
        nhead=8,
        enc_layers=4,
        dec_layers=7,
        dropout=0.1,
        max_len=50,
        pretrained_resnet=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.max_len = max_len

        self.img_enc = ResNet18Tokens(
            d_model=d_model,
            freeze_backbone=True,   # <-- freeze the full ResNet
            freeze_bn=True,
            pretrained=pretrained_resnet
        )

        # Tokens for joints & z
        self.joint_proj = nn.Linear(joint_dim, d_model)
        self.z_proj = nn.Linear(z_dim, d_model)

        # Transformer encoder over observation tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.obs_encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # --- CVAE posterior encoder q(z|a, joints)
        self.act_embed = nn.Linear(action_dim, d_model)
      #  self.step_pos = nn.Embedding(max_len, d_model)
        self.joint_embed_for_q = nn.Linear(joint_dim, d_model)
        q_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.q_encoder = nn.TransformerEncoder(q_layer, num_layers=enc_layers)
        self.to_mu = nn.Linear(d_model, z_dim)
        self.to_logvar = nn.Linear(d_model, z_dim)

        # --- Transformer decoder (autoregressive over K) ---
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
      #  self.tgt_pos = nn.Embedding(max_len, d_model)
        self.out_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, action_dim)
        )

        # CLS token for obs encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _fixed_time_pe(self, K, device, dtype):
        return sinusoid_1d(K, self.d_model, device, dtype)

    def infer_posterior(self, action_seq, joints, mask):

        B, K, _ = action_seq.shape
        a_tok = self.act_embed(action_seq)  # (B,K,d)
       # steps = torch.arange(K, device=action_seq.device).unsqueeze(0).expand(B, K)
        #a_tok = a_tok + self.step_pos(steps)
        steps_pe = self._fixed_time_pe(K, action_seq.device, a_tok.dtype).unsqueeze(0).expand(B, K, -1)
        a_tok  = a_tok + steps_pe
        joints_tok = self.joint_embed_for_q(joints).unsqueeze(1)  # (B,1,d)
        seq = torch.cat([joints_tok, a_tok], dim=1)  # (B,1+K,d)

        # key padding: 0->keep, 1->mask
        key_pad = torch.cat(
            [torch.zeros(B, 1, device=mask.device), 1.0 - mask], dim=1
        ).bool()
        enc = self.q_encoder(seq, src_key_padding_mask=key_pad)  # (B,1+K,d)
        cls_feat = enc[:, 0]  # (B,d)
        mu = self.to_mu(cls_feat)
        logvar = self.to_logvar(cls_feat)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return mu, logvar, z

    def to_device(self, device: str) -> Any:
        """
        Move the underlying policy to a device and return self.
        Args:
            device: Target device identifier (e.g., "cuda", "cpu").

        Returns:
            PiZeroNet: This instance, for method chaining.

        """
        self.to(device)

    def reset(self):
        pass

    def set_eval_mode(self) -> None:
        """
        Set the underlying policy to evaluation mode.
        """

        self.eval()

    def set_train_mode(self) -> None:
        """
        Set the underlying policy to training mode.
        """
        self.train()

    def predict(self):
        pass

    # ----- Build observation memory -----
    def build_memory(self, image, joints, z):
        """
        image:  (B,3,H,W)
        joints: (B,J) or (B,1,J)  <-- we normalize this
        z:      (B, z_dim)
        returns: (B, N_ctx, d)
        """
        B = image.size(0)

        # Normalize joints to (B, J)
        if joints.dim() == 3 and joints.size(1) == 1:
            joints = joints.squeeze(1)  # (B, J)
        elif joints.dim() == 1:
            joints = joints.unsqueeze(0)  # (1, J) -> (B, J) if B==1
        elif joints.dim() != 2:
            raise ValueError(
                f"Expected joints to be (B,J) or (B,1,J), got {tuple(joints.shape)}"
            )

        img_tokens = self.img_enc(image)  # (B, hw, d)
        joints_tok = self.joint_proj(joints).unsqueeze(1)  # (B, 1, d)
        z_tok = self.z_proj(z).unsqueeze(1)  # (B, 1, d)
        cls = self.cls_token.expand(B, 1, -1)  # (B, 1, d)

        # sanity checks (optional)
        # assert img_tokens.dim() == 3 and img_tokens.shape[-1] == self.d_model
        # assert cls.shape == z_tok.shape == joints_tok.shape == (B, 1, self.d_model)

        tokens = torch.cat([cls, z_tok, joints_tok, img_tokens], dim=1)  # (B, N, d)
        return self.obs_encoder(tokens)

        # (B, N, d)

    def decode_actions(self, memory, K):
        B = memory.size(0)
        #steps = torch.arange(K, device=memory.device).unsqueeze(0).expand(B, K)
        steps_pe = self._fixed_time_pe(K, memory.device, memory.dtype).unsqueeze(0).expand(B, K, -1)
        #tgt = self.tgt_pos(steps)  # (B,K,d)
        tgt = steps_pe
        causal = torch.triu(torch.ones(K, K, device=memory.device), diagonal=1).bool()
        out = self.decoder(tgt, memory, tgt_mask=causal)  # (B,K,d)
        return self.out_head(out)  # (B,K,A)

    def forward(self, image, joints, action_seq, mask):
        """
        image:      (B,3,H,W)
        joints:     (B,J)
        action_seq: (B,K,A)
        mask:       (B,K) float {0,1}
        """
        mu, logvar, z = self.infer_posterior(action_seq, joints, mask)
        memory = self.build_memory(image, joints, z)
        K = action_seq.size(1)
        pred = self.decode_actions(memory, K)
        return pred, mu, logvar


def masked_l1(pred, target, mask):
    diff = (pred - target).abs()  # (B,K,A)
    m = mask.unsqueeze(-1)
    num = (diff * m).sum()
    den = (m.sum() * pred.size(-1)).clamp_min(1.0)
    return num / den


def kl_loss(mu, logvar):
    # KL(q||p), p=N(0,I)
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)  # (B,)
