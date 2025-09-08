from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch

from ark_ml.arkml.nodes.policy_node import PolicyNode


class DiffusionPolicyNode(PolicyNode):
    def __init__(self, model, num_diffusion_iters=100, pred_horizon=16, action_dim=None, device="cuda"):
        super().__init__(model, device)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        self.num_diffusion_iters = num_diffusion_iters
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim if action_dim is not None else getattr(model, 'input_dim', 8)

    def prepare_input(self, obs_history):
        """Flatten dict observations into numeric vectors for diffusion policy.

        Expected keys in each obs dict from RobotEnv.observation_unpacking:
        - "cube": (x,y,z)
        - "target": (x,y,z)
        - "gripper": [g]
        - "franka_ee": ((x,y,z), (qx,qy,qz,qw))
        - "images": (rgb, depth)  # ignored

        Returns numpy array of shape (H, 10): [cube(3), target(3), gripper(1), ee_xyz(3)].
        """
        import numpy as _np

        def flatten_one(o: dict):
            cube = _np.asarray(o.get("cube", [0, 0, 0]), dtype=_np.float32).reshape(-1)[:3]
            target = _np.asarray(o.get("target", [0, 0, 0]), dtype=_np.float32).reshape(-1)[:3]
            grip_list = o.get("gripper", [0.0])
            gripper = _np.asarray(grip_list, dtype=_np.float32).reshape(-1)[:1]
            ee_pos, _ee_quat = o.get("franka_ee", ([0, 0, 0], [0, 0, 0, 1]))
            ee = _np.asarray(ee_pos, dtype=_np.float32).reshape(-1)[:3]
            vec = _np.concatenate([cube, target, gripper, ee], axis=0)
            # Ensure length 10
            if vec.shape[0] != 10:
                if vec.shape[0] > 10:
                    vec = vec[:10]
                else:
                    vec = _np.pad(vec, (0, 10 - vec.shape[0]))
            return vec

        if isinstance(obs_history, _np.ndarray):
            return obs_history

        first = obs_history[0]
        if isinstance(first, dict):
            return _np.stack([flatten_one(o) for o in obs_history])
        else:
            # Already numeric sequence
            return _np.stack(obs_history)

    def predict(self, obs_seq):
        # Ensure tensor on correct device
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
        else:
            obs_seq = obs_seq.to(self.device, dtype=torch.float32)

        # (H, obs_dim) -> (1, H*obs_dim)
        obs_cond = obs_seq.unsqueeze(0).flatten(start_dim=1)

        # Initialize sample (B=1, T=pred_horizon, C=action_dim)
        action = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)
        self.scheduler.set_timesteps(self.num_diffusion_iters)

        with torch.no_grad():
            for t in self.scheduler.timesteps:
                noise_pred = self.model(sample=action, timestep=t, global_cond=obs_cond)
                action = self.scheduler.step(model_output=noise_pred, timestep=t, sample=action).prev_sample

        # Return as [T, action_dim]
        return action.squeeze(0).to('cpu').numpy()

    # Backward-compat alias
    def infer(self, obs_seq):
        return self.predict(obs_seq)
