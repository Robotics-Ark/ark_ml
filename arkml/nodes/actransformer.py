import numpy as np
import torch
from torchvision import transforms as T
from typing_extensions import overload
from arktypes.utils import pack
from arktypes import task_space_command_t, string_t

from arkml.algos.ACTransformer.models import ACT
from arkml.core.policy_node import PolicyNode
from typing import Any
from arkml.utils.franka_utils import observation_unpacking, action_packing

import numpy as np
from PIL import Image

class TemporalEnsembler:

    def __init__(self, K, action_dim):
        self.K = K
        self.action_dim = action_dim
        self.sum_buf = np.zeros((K, action_dim), dtype=np.float32)
        self.cnt_buf = np.zeros((K,), dtype=np.float32)

    def step_and_get(self, new_chunk, stride=1):
        K = self.K
        # Shift left
        if stride > 0:
            self.sum_buf[:-stride] = self.sum_buf[stride:]
            self.cnt_buf[:-stride] = self.cnt_buf[stride:]
            self.sum_buf[-stride:] = 0.0
            self.cnt_buf[-stride:] = 0.0

        self.sum_buf[:K] += new_chunk
        self.cnt_buf[:K] += 1.0

        smoothed = self.sum_buf / np.maximum(self.cnt_buf[:, None], 1.0)
        return smoothed[:stride]


class ActPolicyNode(PolicyNode):
    def __init__(
        self,
        model_cfg,
        device="cpu",
        chunk_size=50,
        action_stride=8,
        image_size=256,
    ):
        """
        Returns `actions_to_exec` from predict()
        """
        policy = ACT(
            joint_dim=10,
            action_dim=8,
            z_dim=32,
            d_model=512,
            ffn_dim=3200,
            nhead=8,
            enc_layers=4,
            dec_layers=7,
            dropout=0.1,
            max_len=256,
        )

        super().__init__(
            policy=policy,
            device=device,
            policy_name=model_cfg.policy_node_name,
            observation_unpacking=observation_unpacking,
            action_packing=action_packing,
            stepper_frequency=model_cfg.stepper_frequency,
            global_config=model_cfg.global_config,
        )
        CKPT_PATH = model_cfg.checkpoint
        ckpt = torch.load(CKPT_PATH, map_location=device)
        policy.load_state_dict(ckpt["model"])
        self.stepper_frequency = model_cfg.stepper_frequency

        self.chunk_size = int(chunk_size)
        self.action_stride = int(action_stride)
        self.image_size = int(image_size)
        self.action_dim = int(model_cfg.action_dim)
        self.device = device

        self.to_tensor = T.Compose(
            [
                T.ToTensor(),
                T.Resize((self.image_size, self.image_size), antialias=True),
            ]
        )

        self.ensembler = TemporalEnsembler(self.chunk_size, self.action_dim)

    # def reset(self):
    #     """Call at the start of an episode"""
    #     self.ensembler.sum_buf[:] = 0.0
    #     self.ensembler.cnt_buf[:] = 0.0

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
        img = torch.from_numpy(ob["images"][0].copy()).permute(2, 0, 1)  # (C, H, W)
        img = img.float().div(255.0).unsqueeze(0)  # (1, C, H, W)

        # ---- State ----
        state = torch.from_numpy(ob["state"]).float().unsqueeze(0)  # (1, D)

        return {
            "image": img,
            "state": state,
        }

    @torch.no_grad()
    def _predict_chunk(self, image_np, joints_t, K):

        z_dim = getattr(self.policy, "z_dim", 32)
        z_zero = torch.zeros((1, z_dim), dtype=torch.float32, device=self.device)

        if image_np.ndim == 3:
            img_t = image_np.unsqueeze(0) # -> (1,3,H,W)
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
            obs['image'], obs['state'], self.chunk_size
        )  # (K, action_dim)

        actions_to_exec = self.ensembler.step_and_get(
            new_chunk=chunk_pred, stride=self.action_stride
        )
        return actions_to_exec[-1]

    def publish_action(self, action):

        if action is None or action.shape[0] < 8:
            return

        xyz = np.asarray(action[:3], dtype=np.float32)
        quat = np.asarray(action[3:7], dtype=np.float32)
        grip = float(action[7])
        msg = pack.task_space_command("next_action", xyz, quat, grip)
        self.pub.publish(msg)
