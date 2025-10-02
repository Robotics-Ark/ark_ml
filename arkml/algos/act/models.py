from typing import Any

import torch
import torch.nn as nn
from arkml.core.policy import BasePolicy
from arkml.core.registry import MODELS
from torchvision.models import resnet18, ResNet18_Weights


def sinusoid_1d(
        length: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate 1D sinusoidal positional encodings.

    Args:
        length: Sequence length (L).
        dim: Embedding dimension (D).
        device: Torch device for the tensor.
        dtype: Tensor data type (default: torch.float32).

    Returns:
        Tensor of shape (L, D) containing sinusoidal encodings.
    """
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)  # (L,1)
    i = torch.arange(dim, device=device, dtype=dtype).unsqueeze(0)  # (1,D)
    angle_rates = torch.pow(
        torch.tensor(10000.0, device=device, dtype=dtype),
        -(2 * (i // 2)) / dim
    )
    angles = pos * angle_rates  # (L,D)
    pe = torch.zeros((length, dim), device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe  # (L,D)


def sinusoid_2d(
        h: int,
        w: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate 2D sinusoidal positional encodings.

    Args:
        h: Height (H).
        w: Width (W).
        dim: Embedding dimension (D), must be even.
        device: Torch device for the tensor.
        dtype: Tensor data type (default: torch.float32).

    Returns:
        Tensor of shape (H*W, D) containing sinusoidal encodings.
    """
    assert dim % 2 == 0, "dim must be even for 2D sinusoidal encoding."
    pe_h = sinusoid_1d(h, dim // 2, device, dtype)[:, None, :].repeat(1, w, 1)  # (H,W,D/2)
    pe_w = sinusoid_1d(w, dim // 2, device, dtype)[None, :, :].repeat(h, 1, 1)  # (H,W,D/2)
    pe = torch.cat([pe_h, pe_w], dim=-1)  # (H,W,D)
    return pe.view(h * w, dim)  # (HW,D)


class ResNet18Tokens(nn.Module):
    """
    ResNet-18 backbone wrapper that outputs token embeddings with 2D sinusoidal positional encodings.

    This module uses a ResNet-18 backbone to extract image features, projects them to
    a desired embedding dimension (`d_model`), flattens them into a sequence of tokens,
    and adds sinusoidal positional encodings.

    Args:
        d_model: Output embedding dimension for each token (default: 512).
        freeze_backbone: If True, freeze all backbone parameters and keep in eval mode.
        freeze_bn: If True, set BatchNorm layers to eval and freeze their parameters.
        pretrained: If True, load ImageNet-pretrained ResNet-18 weights.

    Shape:
        - Input: (B, 3, H, W)  
        - Output: (B, H*W, d_model)
    """

    def __init__(self,
                 d_model: int = 512,
                 freeze_backbone: bool = True,
                 freeze_bn: bool = True,
                 pretrained: bool = True, ):
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
        """Set all BatchNorm layers to eval mode and freeze parameters."""
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
                for p in mod.parameters():
                    p.requires_grad = False

    # Keep BN/backbone in eval even if model.train() is called
    def train(self, mode: bool = True):
        """
         Override train() to keep backbone frozen in eval mode if required.

         Args:
             mode: Training mode (True/False).

         """
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
                feat = self.stem(x)  # (B,512,h,w)
        else:
            feat = self.stem(x)

        feat = self.proj(feat)  # (B,d,h,w)
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
    """
    Action-Chunk Transformer policy with a ResNet-18 image encoder and a CVAE-style
    latent posterior over actions.

    The model:
      1) Encodes image -> tokens (with 2D sinusoidal PE) via `ResNet18Tokens`.
      2) Infers a latent z ~ q(z|a, joints) using a Transformer encoder.
      3) Builds an observation memory from (CLS, z, joints, image tokens).
      4) Decodes an autoregressive action sequence with a Transformer decoder.

    Args:
        joint_dim: Size of the joints/state vector (J).
        action_dim: Action dimension per step (A).
        z_dim: Latent dimension for z.
        d_model: Transformer embedding size.
        ffn_dim: Feed-forward dim in encoder/decoder blocks.
        nhead: Attention heads.
        enc_layers: Number of encoder layers (for obs/q encoders).
        dec_layers: Number of decoder layers.
        dropout: Dropout probability in Transformer layers.
        max_len: Max target length (kept for future embeddings).
        pretrained_resnet: If True, use ImageNet-pretrained ResNet-18.
    """

    def __init__(
            self,
            joint_dim: int = 9,
            action_dim: int = 8,
            z_dim: int = 32,
            d_model: int = 512,
            ffn_dim: int = 3200,
            nhead: int = 8,
            enc_layers: int = 4,
            dec_layers: int = 7,
            dropout: float = 0.1,
            max_len: int = 50,
            pretrained_resnet: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.max_len = max_len

        self.img_enc = ResNet18Tokens(
            d_model=d_model,
            freeze_backbone=True,  # <-- freeze the full ResNet
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
        """Initialize module parameters
        * Linear layers: truncated normal for weights (std=0.02), zeros for bias.
        * Conv2d: Kaiming normal for weights (ReLU nonlinearity), zeros for bias.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _fixed_time_pe(self, K: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        """Create fixed sinusoidal positional encodings for K time steps.

        Args:
        K (int): Target sequence length (number of time steps).
        device (torch.device | str): Target device for the returned tensor.
        dtype (torch.dtype): Desired dtype for the returned tensor.


        Returns:
        torch.Tensor: Positional encodings of shape ``(K, d_model)``.
        """
        return sinusoid_1d(K, self.d_model, device, dtype)

    def infer_posterior(self, action_seq: torch.Tensor, joints: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the CVAE approximate posterior ``q(z | a, joints)``.


        The action sequence is embedded with fixed sinusoidal time encodings and
        concatenated with an embedded joint token. A Transformer encoder produces
        a sequence representation whose first token ("CLS") summarizes the input.
        That representation is projected to the mean and log-variance of a
        diagonal Gaussian, and a reparameterized sample :math:`z` is drawn.


        Args:
        action_seq (torch.Tensor): Action sequence of shape ``(B, K, A)``.
        joints (torch.Tensor): Joint/state vector of shape ``(B, J)`` or
        ``(B, 1, J)``.
        mask (torch.Tensor): Float key-padding mask of shape ``(B, K)`` with
        values ``1`` for valid/time steps to keep and ``0`` for padded
        steps.


        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ``(mu, logvar, z)`` where each has shape ``(B, z_dim)`` and ``z``
        is a reparameterized sample.
        """

        B, K, _ = action_seq.shape
        a_tok = self.act_embed(action_seq)  # (B,K,d)
        # steps = torch.arange(K, device=action_seq.device).unsqueeze(0).expand(B, K)
        # a_tok = a_tok + self.step_pos(steps)
        steps_pe = self._fixed_time_pe(K, action_seq.device, a_tok.dtype).unsqueeze(0).expand(B, K, -1)
        a_tok = a_tok + steps_pe
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
    def build_memory(self, image: torch.Tensor, joints: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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

    def decode_actions(self, memory: torch.Tensor, K: int) -> torch.Tensor:
        """Decode an autoregressive sequence of actions.


        Args:
        memory (torch.Tensor): Encoder memory of shape ``(B, N_ctx, d_model)``.
        K (int): Number of target action steps to predict.


        Returns:
        torch.Tensor: Predicted action sequence of shape ``(B, K, A)``.
        """
        B = memory.size(0)
        # steps = torch.arange(K, device=memory.device).unsqueeze(0).expand(B, K)
        steps_pe = self._fixed_time_pe(K, memory.device, memory.dtype).unsqueeze(0).expand(B, K, -1)
        # tgt = self.tgt_pos(steps)  # (B,K,d)
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
