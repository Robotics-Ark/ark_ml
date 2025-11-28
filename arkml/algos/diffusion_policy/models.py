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
import torchvision.transforms as T


def _get_output_shape(
    module: nn.Module, input_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Run a dummy tensor through a module to infer the output shape.

    Args:
        module: The module to probe.
        input_shape: Shape tuple including batch dim (e.g., (1, C, H, W)).
    Returns:
        The output tensor shape as a tuple.
    """
    with torch.no_grad():
        dummy = torch.zeros(*input_shape)
        return tuple(module(dummy).shape)


def _replace_submodules(root_module: nn.Module, predicate, func) -> nn.Module:
    """
    Recursively replace submodules that match ``predicate`` using ``func``.

    This is used to swap batch norms for group norms when requested.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


class SpatialSoftmax(nn.Module):
    """
    Spatial soft-argmax over CNN feature maps producing keypoint coordinates.

    Adapted from Diffusion Policy's vision encoder. Given feature maps (B, C, H, W),
    this outputs (B, K, 2) keypoints where K is either C or ``num_kp``.
    """

    def __init__(self, input_shape: tuple[int, int, int], num_kp: int | None = None):
        super().__init__()
        c, h, w = input_shape
        self._in_h, self._in_w = h, w
        if num_kp is not None:
            self.conv = nn.Conv2d(c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.conv = None
            self._out_c = c

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, w),
            torch.linspace(-1.0, 1.0, h),
            indexing="xy",
        )
        pos = torch.stack([pos_x, pos_y], dim=-1).reshape(h * w, 2)
        self.register_buffer("pos_grid", pos.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            x = self.conv(x)
        b, c, h, w = x.shape
        x = x.reshape(b * c, h * w)
        attn = F.softmax(x, dim=-1)
        expected_xy = attn @ self.pos_grid  # (b*c, 2)
        return expected_xy.view(b, self._out_c, 2)


def get_groupnorm(num_channels: int, preferred_groups: int = 32) -> nn.Module:
    """Return a GroupNorm layer with a valid group count for the channel size."""
    groups = min(preferred_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    groups = max(1, groups)
    return nn.GroupNorm(groups, num_channels)


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
    ResNet vision encoder with spatial softmax pooling (Diffusion Policy style).
    """

    def __init__(
        self,
        out_dim: int = 256,
        freeze_backbone: bool = True,
        trainable_layers: tuple[str, ...] | None = ("layer4",),
        bn_eval: bool = True,
        backbone: str = "resnet18",
        crop_shape: tuple[int, int] | None = None,
        crop_is_random: bool = True,
        use_group_norm: bool = True,
        spatial_softmax_num_keypoints: int = 32,
        input_shape: tuple[int, int, int] | None = None,
        pretrained_backbone_weights: bool = True,
    ):
        super().__init__()
        self._bn_eval = bool(bn_eval)
        self.do_crop = crop_shape is not None
        if self.do_crop:
            self.center_crop = T.CenterCrop(crop_shape)
            self.maybe_random_crop = (
                T.RandomCrop(crop_shape) if crop_is_random else self.center_crop
            )
        else:
            self.center_crop = None
            self.maybe_random_crop = None

        bb_name = (backbone or "resnet18").lower()
        pretrained = bool(pretrained_backbone_weights)
        if bb_name == "resnet50":
            from torchvision.models import ResNet50_Weights, resnet50

            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            bb = resnet50(weights=weights)
        else:
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            bb = resnet18(weights=weights)
        backbone_body = nn.Sequential(*(list(bb.children())[:-2]))

        if use_group_norm:
            if pretrained:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            backbone_body = _replace_submodules(
                root_module=backbone_body,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=max(1, x.num_features // 16), num_channels=x.num_features
                ),
            )

        # Infer feature map shape for spatial softmax pooling.
        c, h, w = input_shape or (
            3,
            *(crop_shape if crop_shape is not None else (224, 224)),
        )
        dummy_shape = (1, c, h, w)
        feature_map_shape = _get_output_shape(backbone_body, dummy_shape)[1:]

        self.backbone = backbone_body
        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=spatial_softmax_num_keypoints
        )
        self.feature_dim = spatial_softmax_num_keypoints * 2
        self.head = nn.Sequential(nn.Linear(self.feature_dim, out_dim), nn.ReLU())

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into a compact feature vector."""
        if self.do_crop and self.maybe_random_crop is not None:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        feats = self.backbone(x)
        feats = torch.flatten(self.pool(feats), start_dim=1)
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


class PastActionEncoder(nn.Module):
    """
    MLP encoder for a short history of past actions.
    """

    def __init__(self, action_dim: int, horizon: int, out_dim: int = 128):
        super().__init__()
        self.horizon = max(1, int(horizon))
        self.out_dim = out_dim
        in_dim = action_dim * self.horizon
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Mish(),
            nn.Linear(out_dim, out_dim),
            nn.Mish(),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, T, action_dim) or (B, action_dim * T) tensor.
        """
        if actions.dim() == 3:
            if actions.shape[1] >= self.horizon:
                actions = actions[:, -self.horizon :]
            else:
                # right-pad with the last action if horizon is longer than provided history
                pad_len = self.horizon - actions.shape[1]
                pad = actions[:, -1:].repeat(1, pad_len, 1)
                actions = torch.cat([actions, pad], dim=1)
            actions = actions.reshape(actions.size(0), -1)
        return self.net(actions)


class DiffusionConv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(
        self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet-style 1D conv block with FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = True,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(
            in_channels, out_channels, kernel_size, n_groups=n_groups
        )
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = DiffusionConv1dBlock(
            out_channels, out_channels, kernel_size, n_groups=n_groups
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


class UNet1D(nn.Module):
    """
    FiLM-conditioned 1D UNet for diffusion over action sequences (LeRobot-style).
    """

    def __init__(
        self,
        action_dim: int = 7,
        horizon: int = 16,
        down_dims: tuple[int, ...] = (256, 512, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 128,
        global_cond_dim: int = 512,
        use_film_scale_modulation: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon

        downsampling_factor = 2 ** len(down_dims)
        if horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be divisible by the total downsampling factor "
                f"({downsampling_factor}). Got horizon={horizon} and down_dims={down_dims}."
            )

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        cond_dim = diffusion_step_embed_dim + global_cond_dim
        in_out = [(action_dim, down_dims[0])] + list(
            zip(down_dims[:-1], down_dims[1:], strict=True)
        )
        common_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "use_film_scale_modulation": use_film_scale_modulation,
        }

        self.down_modules = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(
                            dim_in, dim_out, **common_kwargs
                        ),
                        DiffusionConditionalResidualBlock1d(
                            dim_out, dim_out, **common_kwargs
                        ),
                        (
                            nn.Conv1d(dim_out, dim_out, 3, 2, 1)
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    down_dims[-1], down_dims[-1], **common_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    down_dims[-1], down_dims[-1], **common_kwargs
                ),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for idx, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = idx >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(
                            dim_in * 2, dim_out, **common_kwargs
                        ),
                        DiffusionConditionalResidualBlock1d(
                            dim_out, dim_out, **common_kwargs
                        ),
                        (
                            nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1)
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(
                down_dims[0], down_dims[0], kernel_size=kernel_size, n_groups=n_groups
            ),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: (B, T, action_dim)
            timesteps: (B,) diffusion steps
            global_cond: (B, global_cond_dim)
        Returns:
            Tensor shaped (B, T, action_dim) with predicted noise/sample.
        """
        x = noisy_actions.permute(0, 2, 1)
        t_embed = self.diffusion_step_encoder(timesteps)
        global_feature = torch.cat([t_embed, global_cond], dim=-1)

        skip_features: list[torch.Tensor] = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            skip_features.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, global_feature)

        for res1, res2, upsample in self.up_modules:
            x = torch.cat((x, skip_features.pop()), dim=1)
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.permute(0, 2, 1).contiguous()


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

    Combines vision/state encoders with a FiLM-conditioned 1D UNet (matching
    the LeRobot Diffusion Policy architecture) and uses a DDPM scheduler for
    training and sampling.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 8,
        obs_horizon: int = 1,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        image_emb_dim: int = 256,
        state_emb_dim: int = 128,
        past_action_emb_dim: int = 128,
        diffusion_steps: int = 100,
        eps: float = 1e-6,
        # vision encoder controls
        image_freeze_backbone: bool = True,
        image_trainable_layers: tuple[str, ...] | None = ("layer4",),
        image_bn_eval: bool = True,
        image_backbone: str = "resnet18",
        crop_shape: tuple[int, int] | None = None,
        crop_is_random: bool = True,
        use_group_norm: bool = True,
        spatial_softmax_num_keypoints: int = 32,
        pretrained_backbone_weights: bool = True,
        # diffusion UNet controls
        down_dims: tuple[int, ...] = (256, 512, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 128,
        use_film_scale_modulation: bool = True,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        num_inference_steps: int | None = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.state_key = "state"
        self.action_dim = action_dim
        self.obs_horizon = max(1, int(obs_horizon))
        self.pred_horizon = int(pred_horizon)
        self.action_horizon = int(action_horizon)
        self.eps = eps
        self.past_action_emb_dim = past_action_emb_dim
        self.diffusion_steps = int(diffusion_steps)
        self.num_inference_steps = (
            int(num_inference_steps) if num_inference_steps else self.diffusion_steps
        )
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        cond_dim = image_emb_dim + state_emb_dim + past_action_emb_dim
        self.image_enc = ImageEncoder(
            out_dim=image_emb_dim,
            freeze_backbone=image_freeze_backbone,
            trainable_layers=image_trainable_layers,
            bn_eval=image_bn_eval,
            backbone=image_backbone,
            crop_shape=crop_shape,
            crop_is_random=crop_is_random,
            use_group_norm=use_group_norm,
            spatial_softmax_num_keypoints=spatial_softmax_num_keypoints,
            pretrained_backbone_weights=pretrained_backbone_weights,
            input_shape=None if crop_shape is None else (3, *crop_shape),
        )
        self.joint_enc = JointStateEncoder(
            in_dim=state_dim * self.obs_horizon, out_dim=state_emb_dim
        )
        self.past_action_enc = PastActionEncoder(
            action_dim=action_dim, horizon=action_horizon, out_dim=past_action_emb_dim
        )

        self.unet = UNet1D(
            action_dim=action_dim,
            horizon=self.pred_horizon,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            global_cond_dim=cond_dim,
            use_film_scale_modulation=use_film_scale_modulation,
        )

        # normalization buffers (defaults: identity transform)
        self.register_buffer("action_mean", torch.zeros(1, 1, action_dim))
        self.register_buffer("action_std", torch.ones(1, 1, action_dim))
        self.register_buffer("state_mean", torch.zeros(1, state_dim * self.obs_horizon))
        self.register_buffer("state_std", torch.ones(1, state_dim * self.obs_horizon))

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
            if sm.shape[1] == self.state_dim and self.obs_horizon > 1:
                sm = sm.repeat(1, self.obs_horizon)
            if sm.shape[1] != self.state_dim * self.obs_horizon:
                raise ValueError(
                    f"State mean has shape {sm.shape}, expected ({1}, {self.state_dim * self.obs_horizon})"
                )
            self.state_mean = sm
        if state_std is not None:
            ss = state_std.reshape(1, -1).to(device=device, dtype=torch.float32)
            if ss.shape[1] == self.state_dim and self.obs_horizon > 1:
                ss = ss.repeat(1, self.obs_horizon)
            if ss.shape[1] != self.state_dim * self.obs_horizon:
                raise ValueError(
                    f"State std has shape {ss.shape}, expected ({1}, {self.state_dim * self.obs_horizon})"
                )
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
        if x.dim() == 2:
            if self.obs_horizon > 1:
                x = x.unsqueeze(1).repeat(1, self.obs_horizon, 1)
            x = x.reshape(x.size(0), -1)
        elif x.dim() == 3:
            x = x[:, -self.obs_horizon :, :].reshape(x.size(0), -1)
        else:
            raise ValueError(f"Unsupported state shape {x.shape}")
        return (x - self.state_mean) / (self.state_std + self.eps)

    def _prepare_image(
        self, obs_dict: dict[str, Any]
    ) -> torch.Tensor | list[torch.Tensor] | None:
        """
        Extract and preprocess the latest image from observation dict.

        If images include a time dimension (B, T, C, H, W), selects the last
        frame. Returns a tensor on the current device.
        """
        images_per_camera: list[torch.Tensor] = []
        for key in ArkMLContext.visual_input_features:
            if key not in obs_dict:
                raise KeyError(f"Missing image key '{key}' in observation.")
            images = obs_dict[key]
            if not isinstance(images, torch.Tensor):
                raise ValueError(
                    f"Expected image '{key}' to be torch.Tensor, received {type(images)}."
                )
            # If a history is present (B, T, C, H, W), keep all T frames;
            # otherwise expect (B, C, H, W)
            if images.dim() == 5:
                # keep the most recent obs_horizon frames
                B, T, C, H, W = images.shape
                start = max(0, T - self.obs_horizon)
                frames = images[:, start:, ...].reshape(-1, C, H, W)
                frames = frames.to(self.device, dtype=torch.float32, non_blocking=True)
                frames = torch.clamp(frames, 0.0, 1.0)
                images_per_camera.append(frames)
            else:
                img = images.to(self.device, dtype=torch.float32, non_blocking=True)
                img = torch.clamp(img, 0.0, 1.0)
                images_per_camera.append(img)

        if not images_per_camera:
            return None

        # Normalize to ImageNet stats
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(
            1, 3, 1, 1
        )
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(
            1, 3, 1, 1
        )
        normed = [
            (img - imagenet_mean)
            / imagenet_std  # each is (B[,T],C,H,W) flattened to (N,C,H,W)
            for img in images_per_camera
        ]
        return normed if len(normed) > 1 else normed[0]

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
        return self._norm_state(state.to(torch.float32))

    def _encode_global_condition(self, obs: dict[str, Any]) -> torch.Tensor:
        """
        Encode images, states, and past actions into a single conditioning vector.
        """
        image_inputs = self._prepare_image(obs)
        state_features = self._prepare_state(obs)

        features: list[torch.Tensor] = [self.joint_enc(state_features)]

        if image_inputs is not None:
            if isinstance(image_inputs, list):
                img_embeds: list[torch.Tensor] = []
                for img in image_inputs:
                    img_feat = self.image_enc(img)
                    B = state_features.size(0)
                    if img_feat.size(0) != B:
                        frames = max(1, img_feat.size(0) // B)
                        img_feat = img_feat.view(B, frames, -1).mean(dim=1)
                    img_embeds.append(img_feat)
                img_feat = torch.cat(img_embeds, dim=-1)
            else:
                img_feat = self.image_enc(image_inputs)
                B = state_features.size(0)
                if img_feat.size(0) != B:
                    frames = max(1, img_feat.size(0) // B)
                    img_feat = img_feat.view(B, frames, -1).mean(dim=1)
            features.append(img_feat)

        if "past_actions" in obs:
            past_actions = obs["past_actions"].to(self.device, non_blocking=True)
            past_actions = self._norm_actions(past_actions.to(torch.float32))
            past_feat = self.past_action_enc(past_actions)
        else:
            past_feat = torch.zeros(
                state_features.size(0),
                self.past_action_emb_dim,
                device=self.device,
            )
        features.append(past_feat)

        return torch.cat(features, dim=-1)

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
        norm_actions = self._norm_actions(curr_actions.to(torch.float32))
        # Create noisy actions using forward diffusion in normalized space
        noisy_actions = scheduler.add_noise(norm_actions, noise, timesteps)

        # Predict noise with the model
        cond_vec = self._encode_global_condition(obs=obs)

        noise_pred = self.unet(noisy_actions, timesteps, cond_vec)

        loss = F.mse_loss(noise_pred, noise.to(noise_pred.device))
        return noise_pred, loss

    def sample_actions(self, obs: dict[str, Any], scheduler=None) -> torch.Tensor:
        """
        Sample an action sequence by iterative denoising under the scheduler.
        """
        self.set_eval_mode()
        scheduler = scheduler or self.build_scheduler()
        steps = self.num_inference_steps
        scheduler.set_timesteps(steps)

        # condition from observations
        cond = self._encode_global_condition(obs)
        batch_size = cond.size(0)

        # start from Gaussian noise in normalized action space
        action = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim),
            device=self.device,
        )

        with torch.no_grad():
            for t in scheduler.timesteps:
                # timesteps must be a tensor [B]
                timesteps = torch.full(
                    (batch_size,), t, device=self.device, dtype=torch.long
                )

                # UNet predicts noise (epsilon)
                noise_pred = self.unet(action, timesteps, cond)
                # scheduler updates the action sample
                action = scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=action,
                ).prev_sample
        # denormalize back to env action space
        action = self._denorm_actions(action)
        return action[:, : self.action_horizon]  # (B, action_horizon, action_dim)

    def predict(self, obs: dict[str, Any], **kwargs) -> torch.Tensor:
        actions = self.sample_actions(obs, **kwargs)
        return actions[:, 0].squeeze(0).detach().cpu().numpy()

    def build_scheduler(self):
        """Construct a default DDPM scheduler used for training and sampling."""
        return DDPMScheduler(
            num_train_timesteps=self.diffusion_steps,
            beta_schedule=self.beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            clip_sample=self.clip_sample,
            clip_sample_range=self.clip_sample_range,
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
