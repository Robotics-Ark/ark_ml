from typing import Any, Optional, Sequence, Union

import math
import torch
import torch.nn as nn

from arkml.core.policy import BasePolicy
from arkml.core.registry import MODELS


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        if half_dim == 0:
            raise ValueError("SinusoidalPosEmb dimension must be >= 2")
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(
        self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.view(embed.size(0), 2, self.out_channels, 1)
        scale = embed[:, 0]
        bias = embed[:, 1]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: Optional[Sequence[int]] = None,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        down_dims = list(down_dims or [256, 512, 1024])
        self.input_dim = input_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        self.input_conv = nn.Conv1d(input_dim, down_dims[0], kernel_size=1)

        self.down_modules = nn.ModuleList()
        prev_dim = down_dims[0]
        for idx, dim_out in enumerate(down_dims):
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            prev_dim, dim_out, cond_dim, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim, kernel_size, n_groups
                        ),
                        (
                            Downsample1d(dim_out)
                            if idx < len(down_dims) - 1
                            else nn.Identity()
                        ),
                    ]
                )
            )
            prev_dim = dim_out

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    down_dims[-1], down_dims[-1], cond_dim, kernel_size, n_groups
                ),
                ConditionalResidualBlock1D(
                    down_dims[-1], down_dims[-1], cond_dim, kernel_size, n_groups
                ),
            ]
        )

        self.up_modules = nn.ModuleList()
        for idx, dim_out in enumerate(reversed(down_dims[:-1])):
            dim_in = down_dims[-(idx + 1)]
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in + dim_out, dim_out, cond_dim, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim, kernel_size, n_groups
                        ),
                        Upsample1d(dim_out),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups=n_groups),
            nn.Conv1d(down_dims[0], input_dim, kernel_size=1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = sample.moveaxis(-1, -2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(x.device)
        timestep = timestep.expand(x.size(0))
        time_emb = self.diffusion_step_encoder(timestep)
        if global_cond is None:
            raise ValueError(
                "global_cond must be provided to ConditionalUnet1D forward"
            )
        global_feature = torch.cat([time_emb, global_cond], dim=-1)

        h: list[torch.Tensor] = []
        x = self.input_conv(x)
        for res1, res2, downsample in self.down_modules:
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, global_feature)

        for res1, res2, upsample in self.up_modules:
            skip = h.pop()
            x = torch.cat([x, skip], dim=1)
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            x = upsample(x)

        # Ensure we have the same temporal length as input
        if x.size(-1) > sample.size(-1):
            x = x[..., : sample.size(-1)]
        elif x.size(-1) < sample.size(-1):
            pad = sample.size(-1) - x.size(-1)
            x = torch.nn.functional.pad(x, (0, pad))

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


def _get_activation(name: Optional[str]) -> nn.Module:
    if name is None:
        return nn.Identity()
    name = name.lower()
    if name in {"relu", "relu6"}:
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    if name == "mish":
        return nn.Mish()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unsupported activation '{name}'.")


def _get_norm(norm_type: Optional[str], num_features: int) -> nn.Module:
    if norm_type is None:
        return nn.Identity()
    norm_type = norm_type.lower()
    if norm_type in {"batchnorm", "batch", "bn"}:
        return nn.BatchNorm2d(num_features)
    if norm_type in {"layernorm", "layer", "ln"}:
        return nn.GroupNorm(1, num_features)
    if norm_type in {"groupnorm", "group", "gn"}:
        groups = 32 if num_features % 32 == 0 else 8
        return nn.GroupNorm(groups, num_features)
    raise ValueError(f"Unsupported normalization '{norm_type}'.")


class TimeDistributedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[Sequence[int]] = None,
        output_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        dims = [input_dim]
        hidden_dims = list(hidden_dims or [])
        dims.extend(hidden_dims)
        if output_dim is not None:
            dims.append(output_dim)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(_get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])
        x = self.net(x)
        return x.reshape(*original_shape[:-1], self.output_dim)


class VisualEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: Sequence[int],
        kernel_sizes: Union[int, Sequence[int]] = 3,
        strides: Union[int, Sequence[int]] = 2,
        activation: str = "silu",
        norm: Optional[str] = "layernorm",
        dropout: float = 0.0,
        global_pool: bool = True,
    ):
        super().__init__()
        channels = list(conv_channels)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(channels)
        if isinstance(strides, int):
            strides = [strides] * len(channels)
        if len(kernel_sizes) != len(channels) or len(strides) != len(channels):
            raise ValueError("kernel_sizes and strides must match conv_channels length")
        blocks: list[nn.Module] = []
        curr_in = in_channels
        for idx, out_channels in enumerate(channels):
            kernel = kernel_sizes[idx]
            stride = strides[idx]
            padding = kernel // 2
            blocks.append(
                nn.Conv2d(
                    curr_in,
                    out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
            blocks.append(_get_norm(norm, out_channels))
            blocks.append(_get_activation(activation))
            if dropout > 0:
                blocks.append(nn.Dropout2d(dropout))
            curr_in = out_channels
        self.body = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.global_pool = global_pool
        self.pool = nn.AdaptiveAvgPool2d(1) if global_pool else nn.Identity()
        self.output_dim = curr_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = self.pool(x)
        return x.flatten(1)


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_seq_len: int,
        add_time_embedding: bool = True,
    ):
        super().__init__()
        self.add_time_embedding = add_time_embedding
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if add_time_embedding:
            position = torch.arange(max_seq_len).float()
            div_term = torch.exp(
                torch.arange(0, hidden_dim, 2).float()
                * (-math.log(10000.0) / hidden_dim)
            )
            pe = torch.zeros(max_seq_len, hidden_dim)
            pe[:, 0::2] = torch.sin(position[:, None] * div_term)
            pe[:, 1::2] = torch.cos(position[:, None] * div_term)
            self.register_buffer("positional_encoding", pe, persistent=False)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        if self.add_time_embedding:
            pe = self.positional_encoding[: x.size(1)].unsqueeze(0)
            x = x + pe
        return self.encoder(x)


class TemporalGRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[Sequence[int]] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [])
        dims = [input_dim]
        dims.extend(hidden_dims)
        layers: list[nn.Module] = []
        for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = idx == len(dims) - 2
            if not is_last and use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            if not is_last:
                layers.append(_get_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@MODELS.register("DiffusionPolicyModel")
class DiffusionPolicyModel(BasePolicy):
    def __init__(
        self,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        diffusion_steps: int,
        image_dim: tuple = (3, 480, 640),
        state_dim: int = 9, # TODO
        image_encoder: dict | None = None,
        proprio_encoder: dict | None = None,
        fusion: dict | None = None,
        condition_head: dict | None = None,
        unet: dict | None = None,
        temporal_pooling: str = "flatten",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.diffusion_steps = diffusion_steps
        self.image_dim = image_dim
        self.temporal_pooling = temporal_pooling.lower()

        image_cfg = dict(image_encoder or {})
        proprio_cfg = dict(proprio_encoder or {})
        fusion_cfg = dict(fusion or {})
        condition_cfg = dict(condition_head or {})
        unet_cfg = dict(unet or {})

        self.use_images = image_cfg.pop("enabled", True)
        self.image_key = image_cfg.pop("key", "images")
        self.use_state = proprio_cfg.pop("enabled", True)
        self.state_key = proprio_cfg.pop("key", "state")

        self.image_encoder = None
        self.image_proj = None
        image_feature_dim = 0
        if self.use_images:
            conv_channels = image_cfg.pop("conv_channels", [32, 64, 128, 256])
            kernel_sizes = image_cfg.pop("kernel_sizes", [5, 3, 3, 3])
            strides = image_cfg.pop("strides", [2, 2, 2, 2])
            activation = image_cfg.pop("activation", "silu")
            norm = image_cfg.pop("norm", "layernorm")
            dropout = image_cfg.pop("dropout", 0.0)
            global_pool = image_cfg.pop("global_pool", True)
            self.image_encoder = VisualEncoder(
                in_channels=self.image_dim[0],
                conv_channels=conv_channels,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=activation,
                norm=norm,
                dropout=dropout,
                global_pool=global_pool,
            )
            proj_dim = image_cfg.pop("proj_dim", None)
            if proj_dim is not None:
                self.image_proj = nn.Linear(self.image_encoder.output_dim, proj_dim)
                image_feature_dim = proj_dim
            else:
                self.image_proj = nn.Identity()
                image_feature_dim = self.image_encoder.output_dim

        self.state_encoder = None
        state_feature_dim = 0
        if self.use_state:
            hidden_dims = proprio_cfg.pop("hidden_dims", [128, 128])
            output_dim = proprio_cfg.pop("output_dim", None)
            activation = proprio_cfg.pop("activation", "gelu")
            dropout = proprio_cfg.pop("dropout", 0.0)
            use_layer_norm = proprio_cfg.pop("use_layer_norm", False)
            self.state_encoder = TimeDistributedMLP(
                input_dim=self.state_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                activation=activation,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
            state_feature_dim = self.state_encoder.output_dim

        per_step_feature_dim = image_feature_dim + state_feature_dim
        if per_step_feature_dim == 0:
            raise ValueError(
                "At least one of image_encoder or proprio_encoder must be enabled."
            )

        fusion_type = fusion_cfg.pop("type", "transformer").lower()
        if fusion_type == "transformer":
            hidden_dim = fusion_cfg.pop("hidden_dim", 512)
            num_layers = fusion_cfg.pop("num_layers", 2)
            num_heads = fusion_cfg.pop("num_heads", 8)
            dropout = fusion_cfg.pop("dropout", 0.1)
            add_time_embedding = fusion_cfg.pop("add_time_embedding", True)
            self.temporal_encoder = TemporalTransformerEncoder(
                input_dim=per_step_feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=obs_horizon,
                add_time_embedding=add_time_embedding,
            )
            fusion_output_dim = self.temporal_encoder.output_dim
        elif fusion_type == "gru":
            hidden_dim = fusion_cfg.pop("hidden_dim", 512)
            num_layers = fusion_cfg.pop("num_layers", 1)
            bidirectional = fusion_cfg.pop("bidirectional", False)
            dropout = fusion_cfg.pop("dropout", 0.0)
            self.temporal_encoder = TemporalGRUEncoder(
                input_dim=per_step_feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            )
            fusion_output_dim = self.temporal_encoder.output_dim
        elif fusion_type in {"none", "identity"}:
            self.temporal_encoder = None
            fusion_output_dim = per_step_feature_dim
        else:
            raise ValueError(f"Unsupported fusion type '{fusion_type}'.")

        condition_hidden = condition_cfg.pop("hidden_dims", [512])
        condition_activation = condition_cfg.pop("activation", "gelu")
        condition_dropout = condition_cfg.pop("dropout", 0.0)
        condition_layer_norm = condition_cfg.pop("use_layer_norm", True)
        condition_output = condition_cfg.pop("output_dim", 512)
        mlp_dims = list(condition_hidden) + [condition_output]
        self.condition_head = MLP(
            input_dim=self._pooled_dim(fusion_output_dim),
            hidden_dims=mlp_dims,
            activation=condition_activation,
            dropout=condition_dropout,
            use_layer_norm=condition_layer_norm,
        )
        self.global_cond_dim = self.condition_head.output_dim

        unet_cfg.setdefault("input_dim", action_dim)
        unet_cfg.setdefault("global_cond_dim", self.global_cond_dim)
        self.noise_pred_net = ConditionalUnet1D(**unet_cfg)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _pooled_dim(self, per_step_dim: int) -> int:
        if self.temporal_pooling == "flatten":
            return per_step_dim * self.obs_horizon
        if self.temporal_pooling in {"mean", "avg", "average", "last"}:
            return per_step_dim
        raise ValueError(f"Unsupported temporal pooling '{self.temporal_pooling}'.")

    def _to_tensor(self, data: Any) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            tensor = data
        else:
            tensor = torch.as_tensor(data)
        tensor = tensor.to(self.device)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor

    def _prepare_image(self, obs_dict: dict[str, Any]) -> Optional[torch.Tensor]:
        if not self.use_images or self.image_encoder is None:
            return None
        if self.image_key not in obs_dict:
            raise KeyError(f"Missing image key '{self.image_key}' in observation.")
        images = self._to_tensor(obs_dict[self.image_key])
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.dim() != 5:
            raise ValueError(
                f"Expected images in (B, T, C, H, W) or (B, T, H, W, C) format, got {tuple(images.shape)}"
            )
        if images.shape[2] not in {1, 3} and images.shape[-1] in {1, 3}:
            images = images.permute(0, 1, 4, 2, 3)
        b, t, c, h, w = images.shape
        encoded = self.image_encoder(images.view(b * t, c, h, w))
        encoded = self.image_proj(encoded)
        return encoded.view(b, t, -1)

    def _prepare_state(self, obs_dict: dict[str, Any]) -> Optional[torch.Tensor]:
        if not self.use_state or self.state_encoder is None:
            return None
        if self.state_key not in obs_dict:
            raise KeyError(f"Missing state key '{self.state_key}' in observation.")
        state = self._to_tensor(obs_dict[self.state_key])
        if state.dim() == 2:
            state = state.unsqueeze(0)
        if state.dim() != 3:
            raise ValueError(
                f"Expected state in (B, T, D) format, got {tuple(state.shape)}"
            )
        return self.state_encoder(state)

    def encode_observations(
        self, obs: Union[torch.Tensor, dict[str, Any]]
    ) -> torch.Tensor:
        if isinstance(obs, dict):
            image_features = self._prepare_image(obs)
            state_features = self._prepare_state(obs)
            features = []
            if image_features is not None:
                features.append(image_features)
            if state_features is not None:
                features.append(state_features)
            if not features:
                raise ValueError("No features extracted from observations.")
            fused = torch.cat(features, dim=-1)
        else:
            fused = self._to_tensor(obs)
            if fused.dim() == 2:
                fused = fused.unsqueeze(0)
            if fused.dim() != 3:
                raise ValueError(
                    f"Expected observation tensor with shape (B, T, D) or (T, D), got {tuple(fused.shape)}"
                )
        if fused.size(1) != self.obs_horizon:
            raise ValueError(
                f"Observation horizon mismatch. Expected {self.obs_horizon}, received {fused.size(1)}"
            )
        if self.temporal_encoder is not None:
            fused = self.temporal_encoder(fused)
        if self.temporal_pooling == "flatten":
            pooled = fused.reshape(fused.size(0), -1)
        elif self.temporal_pooling in {"mean", "avg", "average"}:
            pooled = fused.mean(dim=1)
        elif self.temporal_pooling == "last":
            pooled = fused[:, -1]
        else:
            raise ValueError(f"Unsupported temporal pooling '{self.temporal_pooling}'.")
        return self.condition_head(pooled)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        *,
        global_cond: Optional[torch.Tensor] = None,
        obs: Optional[Union[torch.Tensor, dict[str, Any]]] = None,
    ) -> torch.Tensor:
        sample = sample.float()
        if global_cond is None:
            if obs is None:
                raise ValueError(
                    "Either global_cond or obs must be provided to forward()."
                )
            global_cond = self.encode_observations(obs)
        return self.noise_pred_net(sample, timestep, global_cond=global_cond)

    def _build_scheduler(self):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        return DDPMScheduler(
            num_train_timesteps=self.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def sample_actions(
        self,
        obs: Union[torch.Tensor, dict[str, Any]],
        *,
        scheduler=None,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        self.set_eval_mode()
        scheduler = scheduler or self._build_scheduler()
        steps = num_inference_steps or self.diffusion_steps
        scheduler.set_timesteps(steps)
        cond = self.encode_observations(obs)
        batch_size = cond.size(0)
        action = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim), device=self.device
        )
        with torch.no_grad():
            for t in scheduler.timesteps:
                noise_pred = self.noise_pred_net(action, t, global_cond=cond)
                action = scheduler.step(noise_pred, t, action).prev_sample
        return action

    def predict(self, obs: dict[str, Any], **kwargs) -> torch.Tensor:
        actions = self.sample_actions(obs, **kwargs)
        return actions[:, 0]

    def reset(self):
        pass

    def to_device(self, device: str):
        return self.to(torch.device(device))

    def set_eval_mode(self):
        self.eval()

    def set_train_mode(self):
        self.train()
