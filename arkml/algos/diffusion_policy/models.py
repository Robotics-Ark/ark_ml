"""
Diffusion policy models and building blocks for sequence/action generation.

This module implements a lightweight 1D UNet-style architecture and helpers
used to model action trajectories as a denoising diffusion process. It includes:

- Sinusoidal time-step embeddings for diffusion schedules.
- A safe GroupNorm factory that adapts group count to channel size.
- 1D residual blocks plus up/downsampling layers for temporal features.
- Encoders for observations (e.g., images, joint states) to condition the model.
- A `UNet1D` backbone that predicts noise for denoising steps.
- `DiffusionPolicyModel` registered with ArkML and integrated with a
  `DDPMScheduler` for training and sampling.

Typical usage registers the model via the ArkML registry (key: "DiffusionPolicyModel"),
trains it with MSE on predicted noise, and samples actions by iteratively
denoising from Gaussian noise under the chosen scheduler.
"""

from __future__ import annotations

from typing import Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from arkml.core.app_context import ArkMLContext
from arkml.core.policy import BasePolicy
from arkml.core.registry import MODELS
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision.models import resnet18

try:
    from torchvision.models import ResNet18_Weights  # torchvision >= 0.13
except Exception:
    ResNet18_Weights = None


class SinusoidalPosEmb(nn.Module):
    """
    Computes sinusoidal positional / time-step embeddings for diffusion models.

    This module maps a scalar timestep `t` (e.g. diffusion step index) into a
    high-dimensional embedding vector using fixed sine and cosine functions of
    different frequencies, similar to the positional encodings introduced in
    the Transformer architecture.

    Args:
        dim (int): Dimension of the output embedding. Must be >= 2.

    Input:
        t (Tensor): Tensor of shape (B,) or (B, 1) containing timesteps
            (either integer indices or continuous values).

    Output:
        Tensor: Sinusoidal embedding of shape (B, dim).
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim < 2:
            raise ValueError("SinusoidalPosEmb dim must be >= 2")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes sinusoidal positional / time-step embeddings for diffusion models.

        Args:
            t (Tensor): Tensor of shape (B,) or (B, 1) containing timesteps
                (either integer indices or continuous values).

        Returns:
            Tensor: Sinusoidal embedding of shape (B, dim).
        """
        # t: (B,) or (B,1) integer timestep or float tensor
        device = t.device
        dtype = torch.float32
        half = self.dim // 2
        emb_scale = math.log(10000.0) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=device, dtype=dtype) * -emb_scale)
        t_float = t.to(dtype).view(-1, 1)
        args = t_float * freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb  # (B, dim)


# Safe GroupNorm factory
def get_groupnorm(num_channels: int, preferred_groups: int = 32) -> nn.Module:
    """
    Create a GroupNorm layer with a valid number of groups for the given channels.

    This utility ensures the chosen number of groups divides `num_channels`.
    It starts from `preferred_groups` (default: 32) and decreases until a valid
    divisor is found, falling back to 1 if necessary.

    Args:
        num_channels (int): Number of feature channels in the input tensor.
        preferred_groups (int, optional): Preferred maximum number of groups.
            Defaults to 32.

    Returns:
        nn.GroupNorm: A GroupNorm layer with `groups` groups and
        `num_channels` channels.

    Example:
        norm = get_groupnorm(64)    # 32 groups of 2 channels
        norm = get_groupnorm(30)    # falls back to 15 groups of 2
        norm = get_groupnorm(7)     # falls back to 1 group (LayerNorm-like)
    """
    groups = min(preferred_groups, num_channels)
    # ensure groups divide channels, lower until divides
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    groups = max(1, groups)
    return nn.GroupNorm(groups, num_channels)


class ZeroConv1d(nn.Module):
    """
    Zero-initialized 1×1 Conv1d layer for ControlNet-style feature injection.

    This layer applies a pointwise 1D convolution (`kernel_size=1`) whose
    weights and bias are initialized to zero. At initialization, the layer
    outputs zeros regardless of the input, ensuring that it does not
    perturb the base network’s activations. During training, the weights
    are updated and the layer gradually learns to inject conditioning
    features (e.g., past action embeddings in Diff-Control).

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.

    Input:
        x (Tensor): Input tensor of shape (B, in_ch, T),
            where B is batch size and T is sequence length.

    Output:
        Tensor: Output tensor of shape (B, out_ch, T).

    Example:
        layer = ZeroConv1d(64, 128)
        x = torch.randn(16, 64, 24)   # (batch, channels, sequence)
        out = layer(x)                # (16, 128, 24), all zeros at init
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualConv1d(nn.Module):
    """
    Two-layer 1D convolutional residual block with GroupNorm and SiLU.

    The block stacks Conv1d->GroupNorm->SiLU twice and adds a skip path.
    The skip either passes the input or projects it with a 1x1 convolution
    when channel counts differ. Padding keeps the temporal dimension unchanged.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int, optional): Convolution kernel size. Defaults to 3.

    Input:
        x (Tensor): Tensor of shape (B, in_ch, T) containing sequence features.

    Output:
        Tensor: Residual features of shape (B, out_ch, T).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.norm1 = get_groupnorm(out_ch)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)
        self.norm2 = get_groupnorm(out_ch)
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv1d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.skip(x))


class Downsample1d(nn.Module):
    """
    Strided 1D convolutional downsampling layer.

    Applies a Conv1d with kernel_size=4, stride=2, and padding=1 to
    reduce the temporal length by roughly half while keeping channels
    unchanged. Learns a low-pass projection that is more stable than
    naive subsampling.

    Args:
        ch (int): Number of input/output channels.

    Input:
        x (Tensor): Input of shape (B, ch, T).

    Output:
        Tensor: Downsampled tensor of shape (B, ch, ceil(T/2)).
    """

    def __init__(self, ch: int):
        super().__init__()
        self.pool = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class Upsample1d(nn.Module):
    """
    Transposed-conv 1D upsampling layer with optional size correction.

    Uses a ConvTranspose1d with kernel_size=4, stride=2, and padding=1 to
    approximately double the temporal length while keeping channel count the
    same. If a specific output length is required, set `target_length` and the
    result is linearly interpolated to match exactly.

    Args:
        ch (int): Number of input/output channels.

    Input:
        x (Tensor): Input of shape (B, ch, T).
        target_length (int, optional): Desired final temporal length.

    Output:
        Tensor: Upsampled tensor of shape (B, ch, ~2T) or (B, ch, target_length).
    """

    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, target_length: int | None = None
    ) -> torch.Tensor:
        x = self.up(x)
        if target_length is not None and x.shape[-1] != target_length:
            x = F.interpolate(x, size=target_length, mode="linear", align_corners=True)
        return x


class ImageEncoder(nn.Module):
    """
    ResNet18 image encoder producing a fixed-size embedding.

    Uses torchvision's ResNet18 backbone (ImageNet weights when available)
    and maps the 512-d pooled feature to `out_dim`.

    Input:
        x (Tensor): (B, 3, H, W) image tensor normalized to ImageNet stats
            (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

    Output:
        Tensor: Image features of shape (B, out_dim).
    """

    def __init__(
        self,
        out_dim: int = 256,
        freeze_backbone: bool = True,
        trainable_layers: tuple[str, ...] | None = ("layer4",),
        bn_eval: bool = True,
    ):
        super().__init__()
        if "ResNet18_Weights" in globals() and ResNet18_Weights is not None:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(pretrained=True)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(),
        )
        # freezing policy
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if trainable_layers is not None:
                for name in trainable_layers:
                    module = getattr(self.backbone, name, None)
                    if module is not None:
                        for p in module.parameters():
                            p.requires_grad = True
        self._bn_eval = bool(bn_eval)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into a compact feature vector."""
        feats = self.backbone(x)
        return self.head(feats)

    def train(self, mode: bool = True):
        super().train(mode)
        # keep BN layers in eval to avoid running stats drift if requested
        if self._bn_eval:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
        return self


class JointStateEncoder(nn.Module):
    """
    MLP encoder for low-dimensional robot states or joint vectors.

    Applies two Linear+ReLU layers to map `in_dim`-D inputs to `out_dim`-D
    features used to condition the diffusion model.

    Input:
        x (Tensor): (B, in_dim) state tensor.

    Output:
        Tensor: Encoded state of shape (B, out_dim).
    """

    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode joint/state vector to a fixed-size embedding."""
        return self.net(x)


class UNet1D(nn.Module):
    """
    1D UNet backbone for predicting noise over action sequences.

    Consumes a noisy action window and conditioning vector and predicts the
    per-step noise (epsilon). Uses residual Conv1d blocks with down/upsampling,
    skip connections, and MLPs for time/condition embeddings.

    Args:
        action_dim (int): Per-step action dimensionality.
        window_size (int): Temporal length of the input sequence.
        base_ch (int): Base number of channels.
        channel_mult (tuple[int,...]): Channel multipliers per level.
        time_emb_dim (int): Time-step embedding dimension.
        cond_dim (int): Conditioning vector dimension.
    """

    def __init__(
        self,
        action_dim: int = 7,
        window_size: int = 24,
        base_ch: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        time_emb_dim: int = 128,
        cond_dim: int = 512,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.base_ch = base_ch

        # initial proj: actions (B, T, D) -> (B, C, T)
        self.input_proj = nn.Conv1d(action_dim, base_ch, kernel_size=1)

        # time & condition projection
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim + time_emb_dim, base_ch),
            nn.SiLU(),
            nn.Linear(base_ch, base_ch),
        )

        # build encoder blocks
        chs: list[int] = [base_ch * m for m in channel_mult]
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_ch = base_ch
        self.skip_channels = []
        for out_ch in chs:
            self.enc_blocks.append(ResidualConv1d(in_ch, out_ch))
            self.downs.append(Downsample1d(out_ch))
            self.skip_channels.append(out_ch)
            in_ch = out_ch

        # middle
        self.mid1 = ResidualConv1d(in_ch, in_ch * 1)
        self.mid2 = ResidualConv1d(in_ch * 1, in_ch * 1)

        # build decoder blocks (reverse)
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for skip_ch in reversed(self.skip_channels):
            self.ups.append(Upsample1d(in_ch))
            self.dec_blocks.append(ResidualConv1d(in_ch + skip_ch, skip_ch))
            in_ch = skip_ch

        self.final_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_ch, action_dim, kernel_size=1),
        )

    def forward(self, noisy_actions, timesteps, cond_vec, injections=None):
        """
        Predict diffusion noise for a noisy action sequence.

        Args:
            noisy_actions (Tensor): Noisy actions of shape (B, T, D).
            timesteps (Tensor): Time-step indices of shape (B,).
            cond_vec (Tensor): Conditioning vector of shape (B, cond_dim).
            injections (tuple|list|None): Optional per-level feature maps
                (B, C, T) to inject in the encoder.

        Returns:
            Tensor: Predicted noise of shape (B, T, D).
        """
        B, T, D = noisy_actions.shape
        x = noisy_actions.permute(0, 2, 1).contiguous()  # (B, action_dim, T)
        x = self.input_proj(x)

        # project time + condition to base channel size
        t_emb = self.time_mlp(timesteps)  # (B, time_emb_dim)
        cond = torch.cat([t_emb, cond_vec], dim=-1)  # (B, cond_dim+time_dim)
        cond_feat = self.cond_mlp(cond)[:, :, None]  # (B, base_ch, 1)

        # inject condition once at input
        x = x + cond_feat

        skips = []
        # encoder
        for idx, enc in enumerate(self.enc_blocks):
            x = enc(x)
            if (
                injections is not None
                and idx < len(injections)
                and injections[idx] is not None
            ):
                inj = injections[idx]
                if inj.shape[-1] != x.shape[-1]:
                    inj = F.interpolate(
                        inj, size=x.shape[-1], mode="linear", align_corners=True
                    )
                x = x + inj
            skips.append(x)
            x = self.downs[idx](x)

        # mid
        x = self.mid1(x)
        x = self.mid2(x)

        # decoder
        for idx, (up, dec) in enumerate(zip(self.ups, self.dec_blocks)):
            skip = skips.pop()
            x = up(x, target_length=skip.shape[-1])
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.final_conv(x)
        x = x.permute(0, 2, 1).contiguous()  # (B, T, action_dim)

        if x.shape[1] != T:
            x = (
                F.interpolate(
                    x.permute(0, 2, 1), size=T, mode="linear", align_corners=True
                )
                .permute(0, 2, 1)
                .contiguous()
            )
        return x


class TransitionBranch(nn.Module):
    """
    Auxiliary encoder that produces injection features from past actions.

    Mirrors UNet encoder levels and maps outputs through zero-initialized
    1x1 convs to allow safe residual injection without disrupting initial
    behavior.
    """

    def __init__(self, unet: UNet1D):
        super().__init__()
        # Derive channel sizes from enc_blocks
        enc_chs = [b.conv1.out_channels for b in unet.enc_blocks]

        base_in = unet.action_dim
        # create encoder-like blocks that map past action windows -> feature maps
        self.encs: nn.ModuleList = nn.ModuleList()
        for i, ch in enumerate(enc_chs):
            in_ch = base_in if i == 0 else enc_chs[i - 1]
            # use ResidualConv1d blocks for processing
            self.encs.append(ResidualConv1d(in_ch, ch))

        # Create downsample modules to mirror UNet downs
        self.downs: nn.ModuleList = nn.ModuleList([Downsample1d(ch) for ch in enc_chs])

        # Zero convs to map produced features to target encoder feature channels
        # injection targets are the corresponding UNet encoder outputs (same channels)
        self.zero_convs: nn.ModuleList = nn.ModuleList(
            [ZeroConv1d(ch, ch) for ch in enc_chs]
        )

    def forward(self, past_actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Encode past actions into per-level injection tensors.

        Input:
            past_actions (Tensor): (B, T_past, action_dim).

        Returns:
            tuple[Tensor,...]: One tensor per encoder level, each (B, C, T).
        """
        # past_actions: (B, T_past, action_dim)
        x = past_actions.permute(0, 2, 1).contiguous()  # (B, action_dim, T)
        injections: list[torch.Tensor] = []
        for enc, down, zconv in zip(self.encs, self.downs, self.zero_convs):
            x = enc(x)
            # map to injection shape (B, C, T)
            inj = zconv(x)
            injections.append(inj)
            x = down(x)
        # return tuple ordered by encoder level (0..L-1)
        return tuple(injections)


@MODELS.register("DiffusionPolicyModel")
class DiffusionPolicyModel(BasePolicy):
    """
    ArkML policy that learns a diffusion model over action sequences.

    Combines observation encoders with a 1D UNet to predict noise and uses
    a DDPM scheduler for training and sampling.
    """

    def __init__(
        self,
        action_dim: int = 8,
        window_size: int = 24,
        base_ch: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        image_emb_dim: int = 256,
        joint_emb_dim: int = 128,
        time_emb_dim: int = 128,
        diffusion_steps: int = 500,
        eps: float = 1e-6,
        # image encoder freeze controls
        image_freeze_backbone: bool = True,
        image_trainable_layers: tuple[str, ...] | None = ("layer4",),
        image_bn_eval: bool = True,
    ):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.state_key = "state"
        self.action_dim = action_dim
        self.window_size = window_size
        self.eps = eps

        cond_dim = image_emb_dim + joint_emb_dim
        self.image_enc = ImageEncoder(
            out_dim=image_emb_dim,
            freeze_backbone=image_freeze_backbone,
            trainable_layers=image_trainable_layers,
            bn_eval=image_bn_eval,
        )
        self.joint_enc = JointStateEncoder(in_dim=action_dim, out_dim=joint_emb_dim)

        self.unet = UNet1D(
            action_dim=action_dim,
            window_size=window_size,
            base_ch=base_ch,
            channel_mult=channel_mult,
            time_emb_dim=time_emb_dim,
            cond_dim=cond_dim,
        )

        self.transition = TransitionBranch(self.unet)

        # freeze unet weights when training only transition branch
        self.freeze_unet = False

        # normalization buffers (defaults: identity transform)
        self.register_buffer("action_mean", torch.zeros(1, 1, action_dim))
        self.register_buffer("action_std", torch.ones(1, 1, action_dim))
        self.register_buffer("state_mean", torch.zeros(1, action_dim))
        self.register_buffer("state_std", torch.ones(1, action_dim))

    def set_normalization_stats(
        self,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
    ) -> None:
        """
        Set normalization stats for actions and state.

        Shapes:
            action_mean/std: (1, 1, D) or (D,) broadcastable to (B, T, D)
            state_mean/std: (1, D) or (D,) broadcastable to (B, D)
        """
        device = self.device
        if action_mean is not None:
            am = action_mean.reshape(1, 1, -1).to(device=device, dtype=torch.float32)
            self.action_mean = am
        if action_std is not None:
            as_ = action_std.reshape(1, 1, -1).to(device=device, dtype=torch.float32)
            self.action_std = torch.clamp(as_, min=self.eps)
        if state_mean is not None:
            sm = state_mean.reshape(1, -1).to(device=device, dtype=torch.float32)
            self.state_mean = sm
        if state_std is not None:
            ss = state_std.reshape(1, -1).to(device=device, dtype=torch.float32)
            self.state_std = torch.clamp(ss, min=self.eps)

    def _norm_actions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize action vectors using dataset statistics.
        Args:
            x: Action tensor of shape (..., action_dim).

        Returns:
            Normalized actions of the same shape as `x`.

        """
        return (x - self.action_mean) / (self.action_std + self.eps)

    def _denorm_actions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize action vectors back to the original scale.
        Args:
            x: Normalized action tensor of shape (..., action_dim).

        Returns:
            Denormalized actions of the same shape as `x`.
        """
        return x * (self.action_std + self.eps) + self.action_mean

    def _norm_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize state vectors using dataset statistics.
        Args:
            x: State tensor of shape (..., state_dim).

        Returns:
            Normalized states of the same shape as `x`.
        """
        return (x - self.state_mean) / (self.state_std + self.eps)

    def _prepare_image(self, obs_dict: dict[str, Any]) -> torch.Tensor | None:
        """
        Extract and preprocess the latest image from observation dict.

        If images include a time dimension (B, T, C, H, W), selects the last
        frame. Returns a tensor on the current device.
        """
        encoded_per_camera: list[torch.Tensor] = []
        for key in ArkMLContext.visual_input_features:
            if key not in obs_dict:
                raise KeyError(f"Missing image key '{key}' in observation.")
            images = obs_dict[key]
            if not isinstance(images, torch.Tensor):
                raise ValueError(
                    f"Expected image '{key}' to be torch.Tensor, received {type(images)}."
                )
            # TODO here it takes latest image,
            #  if the policy to see a history of frames then encode images separately and fuse it or similar method
            if images.dim() == 5:
                image = images[:, -1]
            else:
                image = images

            image = image.to(self.device, dtype=torch.float32, non_blocking=True)
            # Expect inputs in [0,1]; clamp and normalize to ImageNet stats
            image = torch.clamp(image, 0.0, 1.0)
            imagenet_mean = torch.tensor(
                [0.485, 0.456, 0.406], device=self.device
            ).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(
                1, 3, 1, 1
            )
            image = (image - imagenet_mean) / imagenet_std
            return image

        return torch.cat(encoded_per_camera, dim=-1) if encoded_per_camera else None

    def _prepare_state(self, obs_dict: dict[str, Any]) -> torch.Tensor:
        """
        Extract and preprocess the latest state vector from observation dict.

        If states include a time dimension (B, T, D), selects the last step.
        Returns a tensor on the current device.
        """
        if self.state_key not in obs_dict:
            raise KeyError(f"Missing state key '{self.state_key}' in observation.")
        state = obs_dict[self.state_key].to(self.device, non_blocking=True)
        if not isinstance(state, torch.Tensor):
            raise ValueError(
                f"Expected state to be of type torch.Tensor, but got {type(state)}."
            )
        # TODO here it takes latest state,
        #  if the policy to see a history of states then change accordingly
        if state.dim() == 3:
            state = state[:, -1]

        # normalize state
        state = state.to(torch.float32)
        state = self._norm_state(state)
        return state

    def encode_observations(self, obs: dict[str, Any]) -> torch.Tensor:
        """
        Encode observations into a conditioning vector for the UNet.

        Concatenates image and state encodings into a single tensor
        of shape (B, image_emb_dim + joint_emb_dim).
        """
        if isinstance(obs, dict):
            image_features = self._prepare_image(obs)
            state_features = self._prepare_state(obs)
            img_e = self.image_enc(image_features)  # (B, image_emb_dim)
            j_e = self.joint_enc(state_features)  # (B, joint_emb_dim)
            cond_vec = torch.cat([img_e, j_e], dim=-1)
            return cond_vec
        else:
            raise ValueError(f"Expected observation to be a dict but got {type(obs)}")

    def _predict_noise(self, sample, timesteps, cond, past_actions=None):
        """
        Predict noise using UNet with optional past-action injections.

        Moves inputs to the correct device and forwards through UNet.
        """
        device = self.device

        # ensure everything is on the same device
        sample = sample.to(device)
        timesteps = timesteps.to(device)
        cond = cond.to(device)

        injections = None
        if past_actions is not None:
            past_actions = past_actions.to(device)
            # normalize past actions before computing injections
            past_actions = self._norm_actions(past_actions)
            injections = self.transition(past_actions)
            injections = (
                [inj.to(device) for inj in injections]
                if isinstance(injections, (list, tuple))
                else injections
            )

        return self.unet(sample, timesteps, cond, injections=injections)

    def forward(
        self, obs: dict[str, Any], scheduler
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training step: diffuse actions and predict noise.

        Samples random noise and timesteps, diffuses current actions, encodes
        observations, and computes MSE loss on predicted noise.

        Returns:
            (noise_pred, loss)
        """

        curr_actions = obs["action"].to(self.device, non_blocking=True)
        past_actions = obs["past_actions"].to(self.device, non_blocking=True)
        batch_size = curr_actions.size(0)

        # Sample random Gaussian noise
        noise = torch.randn_like(curr_actions)

        # Sample random timesteps
        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

        # Normalize actions before diffusion
        curr_actions = curr_actions.to(torch.float32)
        norm_actions = self._norm_actions(curr_actions)
        # Create noisy actions using forward diffusion in normalized space
        noisy_actions = scheduler.add_noise(norm_actions, noise, timesteps)

        # Predict noise with the model
        cond_vec = self.encode_observations(obs=obs)

        if self.freeze_unet:
            # disable grad for unet
            for p in self.unet.parameters():
                p.requires_grad = False

        noise_pred = self._predict_noise(
            sample=noisy_actions,
            timesteps=timesteps,
            cond=cond_vec,
            past_actions=past_actions,
        )

        loss = F.mse_loss(noise_pred, noise.to(noise_pred.device))
        return noise_pred, loss

    def sample_actions(self, obs: dict[str, Any], scheduler) -> torch.Tensor:
        """
        Sample an action sequence by iterative denoising under the scheduler.
        """
        self.set_eval_mode()
        steps = self.diffusion_steps
        scheduler.set_timesteps(steps)

        # condition from observations
        cond = self.encode_observations(obs)
        past_actions = obs["past_actions"]
        batch_size = cond.size(0)

        # start from Gaussian noise in normalized action space
        action = torch.randn(
            (batch_size, self.window_size, self.action_dim),
            device=self.device,
        )

        with torch.no_grad():
            for t in scheduler.timesteps:
                # timesteps must be a tensor [B]
                timesteps = torch.full(
                    (batch_size,), t, device=self.device, dtype=torch.long
                )

                # UNet predicts noise (epsilon)
                noise_pred = self._predict_noise(
                    sample=action,
                    timesteps=timesteps,
                    cond=cond,
                    past_actions=past_actions,
                )
                # scheduler updates the action sample
                action = scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=action,
                ).prev_sample
        # denormalize back to env action space
        action = self._denorm_actions(action)
        return action  # (B, pred_horizon, action_dim)

    def predict(self, obs: dict[str, Any], **kwargs) -> torch.Tensor:
        actions = self.sample_actions(obs, **kwargs)
        return actions[:, 0].squeeze(0).detach().cpu().numpy()

    def build_scheduler(self):
        """Construct a default DDPM scheduler used for training and sampling."""
        return DDPMScheduler(
            num_train_timesteps=self.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def reset(self):
        """
        Reset internal policy state.
        """
        pass

    def to_device(self, device: str):
        """
        Move the underlying policy to a device and return self.
        Args:
            device: Target device identifier (e.g., "cuda", "cpu").

        Returns:
            This instance, for method chaining.

        """
        return self.to(torch.device(device))

    def set_eval_mode(self):
        """
        Set the underlying policy to evaluation mode.
        """
        self.eval()

    def set_train_mode(self):
        """
        Set the underlying policy to training mode.
        """
        self.train()

    @property
    def device(self) -> torch.device:
        """
        Device identifier the policy is being on.
        Returns:
            Returns the device identifier.
        """
        return next(self.parameters()).device
