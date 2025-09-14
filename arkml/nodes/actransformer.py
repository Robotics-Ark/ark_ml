import pdb

import numpy as np
import torch
from torchvision import transforms as T
from typing_extensions import overload
from arktypes.utils import pack
from arktypes import task_space_command_t, string_t

from arkml.algos.ACTransformer.models import ACT
from arkml.core.policy_node import PolicyNode

import numpy as np
from PIL import Image

def coerce_len1_len3_image(x):
    """
    Accepts images shaped like:
        - [ [R, G, B] ]   where each channel is HxW (lists or np arrays), or
        - [ [C, H, W] ]   CHW-style as nested lists/np arrays, or
        - already a HWC/CHW array/tensor/PIL.
    Returns a PIL.Image.Image in HWC uint8.
    """
    # Unwrap an outer length-1 container
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]

    # If it's now a length-3 list/tuple, assume channels or CHW
    if isinstance(x, (list, tuple)) and len(x) == 3:
        chans = [np.asarray(c) for c in x]
        # Case A: three HxW channel planes -> stack to HWC
        if all(c.ndim == 2 for c in chans):
            img = np.stack(chans, axis=-1)  # HxWx3
        else:
            # Case B: likely CHW -> convert to HWC
            arr = np.asarray(x)
            if arr.ndim == 3 and arr.shape[0] == 3:  # (3,H,W)
                img = np.transpose(arr, (1, 2, 0))
            else:
                # Fallback: try to interpret directly as HWC
                img = np.asarray(x)
    else:
        # Already array-like; try to make it HWC
        img = np.asarray(x)

    # If grayscale (H,W), promote to 3 channels
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)

    # Ensure uint8 in [0,255]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return Image.fromarray(img)


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
    def __init__(self, model_cfg, device="cpu", chunk_size=50, action_stride=1, image_size=256, global_config=None):
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
            channel_type=string_t,
            message_type=task_space_command_t,
            global_config=global_config,
        )
        CKPT_PATH = model_cfg.checkpoint
        ckpt = torch.load(CKPT_PATH, map_location=device)
        policy.load_state_dict(ckpt["model"])


        self.chunk_size = int(chunk_size)
        self.action_stride = int(action_stride)
        self.image_size = int(image_size)
        self.action_dim = int(model_cfg.action_dim)
        self.device = device

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Resize((self.image_size, self.image_size), antialias=True),
        ])


        self.ensembler = TemporalEnsembler(self.chunk_size, self.action_dim)

    def reset(self):
        """Call at the start of an episode"""
        self.ensembler.sum_buf[:] = 0.0
        self.ensembler.cnt_buf[:] = 0.0

    @staticmethod
    def _build_joint_state(obs):
        # cube(3) + target(3) + gripper(1) + franka_ee(3) = 10
        cube      = np.asarray(obs["cube"]).reshape(-1)
        target    = np.asarray(obs["target"]).reshape(-1)
        gripper   = np.asarray(obs["gripper"]).reshape(-1)
        franka_ee = np.asarray(obs["franka_ee"][0]).reshape(-1)
        vec = np.concatenate([cube, target, gripper, franka_ee], axis=0)
        if vec.shape[0] != 10:
            raise ValueError(f"Expected 10-D joints vector, got {vec.shape}")
        return torch.tensor(vec, dtype=torch.float32)

    @staticmethod
    def _extract_image(obs):
        img = obs["images"]
        if isinstance(img, dict):
            cam_name = sorted(img.keys())[0]
            img = img[cam_name]
        elif isinstance(img, (list, tuple)):
            img = img[0]
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] > 3:
            img = img[..., :3]
        return img

    @torch.no_grad()
    def _predict_chunk(self, image_np, joints_t, K):

        z_dim = getattr(self.policy, "z_dim", 32)
        z_zero = torch.zeros((1, z_dim), dtype=torch.float32, device=self.device)

        # actransformer.py::_predict_chunk

        # image_np is that list-of-len-1 whose first entry is a list-of-len-3
        pil_img = coerce_len1_len3_image(image_np)  # -> PIL.Image (H,W,3), uint8

        # Run your torchvision pipeline (e.g., Resize/ToTensor/Normalize).
        img_t = self.to_tensor(pil_img)  # -> torch.Tensor (3,H,W)

        # Ensure batch dimension since the model expects (1,3,H,W)
        if isinstance(img_t, torch.Tensor) and img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)  # -> (1,3,H,W)

        img_t = img_t.to(self.device)

        j_t   = torch.tensor(joints_t[0]).unsqueeze(0).to(self.device)                  # (1,10)

        memory = self.policy.build_memory(img_t, j_t, z_zero)          # (1, N_ctx, d)
        pred   = self.policy.decode_actions(memory, K)                 # (1,K,action_dim)
        return pred.squeeze(0).cpu().numpy()                          # (K,action_dim)

    @torch.no_grad()
    def predict(self, obs):
        """
        Returns: actions_to_exec (shape: (action_stride, action_dim))
        """
        joints = obs['state']
        image  = obs['image']
        episode_over = obs["episode_over"]

        if episode_over:

            print("Episode overdfbvj")
            self.reset()

            print(self.ensembler.sum_buf[:])
            return None
        else:

            chunk_pred = self._predict_chunk(image, joints, self.chunk_size)  # (K, action_dim)

            actions_to_exec = self.ensembler.step_and_get(
                new_chunk=chunk_pred,
                stride=self.action_stride
            )
            return actions_to_exec[0]

    # TODO implement
    def publish_action(self, action):
        """Pack and publish action to downstream consumers."""
        if action.shape[0] < 8:
            return

        xyz = np.asarray(action[:3], dtype=np.float32)
        quat = np.asarray(action[3:7], dtype=np.float32)
        grip = float(action[7])
        msg = pack.task_space_command("next_action", xyz, quat, grip)
        self.pub.publish(msg)
