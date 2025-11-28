from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from arkml.algos.act.models import ACT
from arkml.core.app_context import ArkMLContext
from arkml.core.policy_node import PolicyNode
from torchvision import transforms as T


class TemporalEnsembler:
    """
      - Maintain per-index weighted sums (numerator) and weights (denominator).
      - On each control step:
          * shift left by `stride`
          * exponentially decay ALL existing contributions by exp(-coeff * stride)
          * add the new chunk with unit weight at every relative index
      - Return the discounted average for the next `stride` steps.
    Note:
      sum_buf -> numerator of weighted avg
      cnt_buf -> denominator (total weight)
    """

    def __init__(self, K: int, action_dim: int, coeff: float = 0.01):
        self.K = int(K)
        self.action_dim = int(action_dim)
        self.coeff = float(coeff)

        # keep original field names for compatibility with your reset()
        self.sum_buf = np.zeros(
            (self.K, self.action_dim), dtype=np.float32
        )  # numerator
        self.cnt_buf = np.zeros((self.K,), dtype=np.float32)  # denominator (weights)

    def step_and_get(
        self,
        new_chunk: npt.NDArray[np.floating],
        stride: int = 1,
    ):
        """
        Update the ensembler with a new chunk and return the next `stride` actions.

        Args:
            new_chunk: Array of predicted actions with shape ``(K, action_dim)``.
                       Extra rows/cols are ignored; if shorter, only the first rows are used.
            stride: Number of steps to advance and emit.

        Returns:
            Discounted-averaged actions of shape ``(stride, action_dim)``.
        """
        K = self.K
        stride = int(stride)

        # 1) shift buffers left by stride
        if stride > 0:
            if stride < K:
                self.sum_buf[:-stride] = self.sum_buf[stride:]
                self.cnt_buf[:-stride] = self.cnt_buf[stride:]
            self.sum_buf[-stride:] = 0.0
            self.cnt_buf[-stride:] = 0.0

        # 2) exponential decay of existing contributions by elapsed steps
        decay = np.exp(-self.coeff * stride).astype(np.float32)
        self.sum_buf *= decay
        self.cnt_buf *= decay

        # 3) add the new chunk with unit weight at all visible indices
        #    (makes the aggregator a discounted average over overlapping chunks)
        self.sum_buf[:K] += new_chunk[:K]
        self.cnt_buf[:K] += 1.0

        # 4) form discounted average; safe divide
        denom = np.maximum(self.cnt_buf[:], 1e-8)[:, None]
        smoothed = self.sum_buf / denom

        # emit the next `stride` actions
        return smoothed[:stride]


class ActPolicyNode(PolicyNode):
    """
    PolicyNode wrapper for the ACT (Action-Chunk Transformer) policy.

    This node loads a trained ACT policy, prepares environment observations,
    runs forward passes, and returns actions for execution.

    Args:
        cfg: Configuration object with fields for model and node setup.
        device: Torch device to run the policy on (default: "cpu").
    """

    def __init__(
        self,
        device="cpu",
    ):
        """
        Returns `actions_to_exec` from predict()
        """
        model_cfg = ArkMLContext.cfg.get("algo").get("model")
        policy = ACT(
            joint_dim=model_cfg.get("joint_dim"),
            action_dim=model_cfg.get("action_dim"),
            z_dim=model_cfg.get("z_dim"),
            d_model=model_cfg.get("d_model"),
            ffn_dim=model_cfg.get("ffn_dim"),
            nhead=model_cfg.get("nhead"),
            enc_layers=model_cfg.get("enc_layers"),
            dec_layers=model_cfg.get("dec_layers"),
            dropout=model_cfg.get("dropout"),
            max_len=model_cfg.get("max_len"),
        )

        super().__init__(
            policy=policy,
            device=device,
            policy_name=ArkMLContext.cfg.get("node_name"),
        )

        CKPT_PATH = model_cfg.get("model_path")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        policy.load_state_dict(ckpt["model_state_dict"])
        self.stepper_frequency = model_cfg.stepper_frequency

        self.chunk_size = int(model_cfg.chunk_size)
        self.action_stride = int(model_cfg.action_stride)
        self.image_size = int(model_cfg.image_size)
        self.action_dim = int(model_cfg.action_dim)
        self.device = device

        self.to_tensor = T.Compose(
            [
                T.ToTensor(),
                T.Resize((self.image_size, self.image_size), antialias=True),
            ]
        )

        self.ensembler = TemporalEnsembler(self.chunk_size, self.action_dim)

    def _on_reset(self):
        """Clear any prefetched actions when an episode ends."""
        self.ensembler.sum_buf[:] = 0.0
        self.ensembler.cnt_buf[:] = 0.0

    def prepare_observation(self, ob: dict[str, Any]):
        """Convert a single raw env observation into a batched policy input.

        Args:
          ob: Single observation dict from the env. Expected to contain keys:
            ``images`` (tuple with RGB as HxWxC),
            ``cube``, ``target``, ``gripper``, and ``franka_ee``.
          task_prompt: Natural language task description to include in the batch.

        Returns:
          A batch dictionary with:
            - ``image``: ``torch.FloatTensor`` of shape ``[1, C, H, W]``.
            - ``state``: ``torch.FloatTensor`` of shape ``[1, D]``.
            - ``task``: ``list[str]`` of length 1.
        """
        # ---- Image ----
        img = torch.from_numpy(ob["sensors::top_camera::rgb"].copy()).permute(
            2, 0, 1
        )  # (C, H, W)
        img = img.float().div(255.0).unsqueeze(0)  # (1, C, H, W)

        # ---- State ----
        state = np.concatenate(
            [
                np.ravel(ob["proprio::pose::position"]),
                np.ravel(ob["proprio::pose::orientation"]),
                np.ravel([ob["proprio::joint_state::position"][-2:]]),
            ]
        )

        state = torch.from_numpy(state).float().unsqueeze(0)  # (1, D)

        return {
            "image": img,
            "state": state,
        }

    @torch.no_grad()
    def _predict_chunk(self, image_np, joints_t, K):
        """
        Predict a chunk of K actions from image and joint inputs.
        """
        z_dim = getattr(self.policy, "z_dim", 32)
        z_zero = torch.zeros((1, z_dim), dtype=torch.float32, device=self.device)

        if image_np.ndim == 3:
            img_t = image_np.unsqueeze(0)  # -> (1,3,H,W)
        else:
            img_t = image_np

        img_t = img_t.to(self.device)

        j_t = torch.tensor(joints_t[0]).unsqueeze(0).to(self.device)  # (1,10)

        memory = self.policy.build_memory(img_t, j_t, z_zero)  # (1, N_ctx, d)
        pred = self.policy.decode_actions(memory, K)  # (1,K,action_dim)
        return pred.squeeze(0).cpu().numpy()  # (K,action_dim)

    @torch.no_grad()
    def predict(self, obs_seq):
        """
        Returns: actions_to_exec (shape: (action_stride, action_dim))
        """

        obs_seq = self.prepare_observation(obs_seq)
        obs = {
            "image": torch.tensor(obs_seq["image"], dtype=torch.float32),
            "state": torch.tensor(obs_seq["state"], dtype=torch.float32),
        }
        chunk_pred = self._predict_chunk(
            obs["image"], obs["state"], self.chunk_size
        )  # (K, action_dim)

        actions_to_exec = self.ensembler.step_and_get(
            new_chunk=chunk_pred, stride=self.action_stride
        )
        return actions_to_exec[-1]
