from collections import deque
from typing import Any
import numpy as np
import torch
from arkml.algos.vla.pi05.models import Pi05Policy
from arkml.core.app_context import ArkMLContext
from arkml.core.policy_node import PolicyNode
from arkml.utils.utils import _image_to_tensor
from arktypes import string_t


class Pi05Node(PolicyNode):
    """
    Policy node for Pi0.5 integration.
    Structurally identical to PiZeroPolicyNode, using Pi05Policy internally.
    """

    def __init__(self, device: str = "cpu", **kwargs):
        """
        Initialize the Pi0.5 policy node.

        Args:
            device: Device to run the model on
        """
        cfg = ArkMLContext.cfg
        model_cfg = cfg.get("algo").get("model")

        policy = Pi05Policy(
            policy_type=model_cfg.get("policy_type"),
            model_path=model_cfg.get("model_path"),
            obs_dim=model_cfg.get("obs_dim"),
            action_dim=model_cfg.get("action_dim"),
            image_dim=model_cfg.get("image_dim"),
            pred_horizon=model_cfg.get("pred_horizon", 1),
        )

        super().__init__(
            policy=policy,
            device=device,
            policy_name=cfg.get("node_name"),
        )

        # Listen to text prompt channel
        channel_name = ArkMLContext.global_config.get("channel", "user_input")
        self.text_input = None
        self.sub = self.create_subscriber(
            channel_name, string_t, self._callback_text_input
        )

        self.policy.to_device(device)
        self.policy.reset()
        self.policy.set_eval_mode()

        self.n_infer_actions = getattr(model_cfg, "pred_horizon", 1)
        self._action_queue: deque[np.ndarray] = deque()

    def _on_reset(self) -> None:
        """
        Policy specific reset function.

        Returns:
            None
        """
        self.policy.reset()

    def predict(self, obs_seq):
        """Compute the action for the given observation batch.

        The expected structure of ``obs_seq`` is dictated by the underlying VLA
        policy (typically a dict with batched tensors for images and state, and
        a list[str] for the task prompt).

        Args:
          obs_seq: Observation input to the policy (dict or tensor as required
            by the wrapped model).

        Returns:
          numpy.ndarray: Action vector for the first batch element.
        """

        obs = self.prepare_observation(obs_seq)

        with torch.no_grad():
            actions = self.policy.predict(obs, n_actions=self.n_infer_actions)
            actions = actions.detach().cpu().numpy()

        return actions[0]

    def prepare_observation(self, ob: dict[str, Any]):
        """Convert a single raw env observation into a batched policy input.

        Args:
          ob: Single observation dict from the env. Expected keys include
            ``state`` and any camera names listed in ``visual_input_features``.

        Returns:
          A batch dictionary with:
            - per-camera image tensors: ``torch.FloatTensor`` of shape ``[1, C, H, W]``.
            - ``state``: ``torch.FloatTensor`` of shape ``[1, D]`` if present.
            - ``task``: ``list[str]`` of length 1 (optional - can be omitted if no language input).
        """
        obs = {}

        # Use provided text input if available, otherwise don't include task key
        # This allows the system to work when language input is not provided by Ark
        if self.text_input is not None and self.text_input.strip() != "":
            obs["task"] = [self.text_input]
        # If no text input, we don't add the task key, and the policy will handle it

        # VALIDATE REQUIRED OBSERVATION KEYS
        # Check for required proprioception data with explicit validation
        required_keys = ["proprio::pose::position", "proprio::pose::orientation", "proprio::joint_state::position"]
        optional_keys = ["sensors::image_top::rgb"]  # Will be handled separately

        # Validate that observation contains at least some expected keys
        available_keys = set(ob.keys())
        required_present = [key for key in required_keys if key in available_keys]

        if not required_present:
            raise ValueError(
                f"Missing required observation keys. Expected at least one of: {required_keys}. "
                f"Available keys: {list(available_keys)}"
            )

        # Extract required data with validation
        position_data = ob.get("proprio::pose::position")
        orientation_data = ob.get("proprio::pose::orientation")
        joint_state_data = ob.get("proprio::joint_state::position")

        # Build state tensor with defensive fallbacks for missing data
        state_components = []

        # Add position data if available, otherwise use zero tensor
        if position_data is not None:
            if not isinstance(position_data, (np.ndarray, list)):
                raise ValueError(f"Expected 'proprio::pose::position' to be array-like, got {type(position_data)}")
            position_data = np.asarray(position_data)
            state_components.append(np.ravel(position_data))
        else:
            # Fallback: use zero tensor of expected size based on model config
            model_cfg = ArkMLContext.cfg.get("algo", {}).get("model", {})
            obs_dim = model_cfg.get("obs_dim", 9)  # Default to 9 if not specified
            # Calculate how many elements we need for position based on expected total
            # For now, assume position is 3 elements (x, y, z)
            state_components.append(np.zeros(3, dtype=np.float32))

        # Add orientation data if available, otherwise use zero tensor
        if orientation_data is not None:
            if not isinstance(orientation_data, (np.ndarray, list)):
                raise ValueError(f"Expected 'proprio::pose::orientation' to be array-like, got {type(orientation_data)}")
            orientation_data = np.asarray(orientation_data)
            state_components.append(np.ravel(orientation_data))
        else:
            # Fallback: assume orientation is 3 elements (roll, pitch, yaw) or 4 (quaternion)
            # Using 3 for now to match the expected total
            state_components.append(np.zeros(3, dtype=np.float32))

        # Add joint state data if available, otherwise use zero tensor
        if joint_state_data is not None:
            if not isinstance(joint_state_data, (np.ndarray, list)):
                raise ValueError(f"Expected 'proprio::joint_state::position' to be array-like, got {type(joint_state_data)}")
            joint_state_data = np.asarray(joint_state_data)
            # Take the last 2 joint positions as in the original code
            if len(joint_state_data) >= 2:
                joint_positions = np.ravel([joint_state_data[-2:]])
            else:
                joint_positions = np.ravel([joint_state_data])
            state_components.append(joint_positions)
        else:
            # Fallback: use 2 zero elements for joint positions
            state_components.append(np.zeros(2, dtype=np.float32))

        # Concatenate all state components
        state = np.concatenate(state_components)
        state = torch.from_numpy(state).float().unsqueeze(0)  # (1, D)
        img = torch.from_numpy(ob["sensors::top_camera::rgb"].copy()).permute(
            2, 0, 1
        )  # (C, H, W)
        img = img.float().div(255.0).unsqueeze(0)  # (1, C, H, W)


        obs["state"] = state

        # Handle image data with defensive access and validation
        # Check for the primary image key first
        primary_image_data = ob.get("sensors::image_top::rgb")

        if primary_image_data is not None:
            # Validate image data format
            if not isinstance(primary_image_data, (np.ndarray, list)):
                raise ValueError(f"Expected 'sensors::image_top::rgb' to be array-like, got {type(primary_image_data)}")
            # Use the available image data
            img = torch.from_numpy(np.asarray(primary_image_data).copy()).permute(2, 0, 1)  # (C, H, W)
            img = img.float().div(255.0).unsqueeze(0)  # (1, C, H, W)
        else:
            # Check if there are any visual input features defined and try to get one
            visual_features = getattr(ArkMLContext, 'visual_input_features', [])
            if visual_features:
                # Try to get the first available visual input
                first_visual_key = visual_features[0] if len(visual_features) > 0 else None
                if first_visual_key and first_visual_key in ob:
                    img_data = ob[first_visual_key]
                    if not isinstance(img_data, (np.ndarray, list)):
                        raise ValueError(f"Expected visual input '{first_visual_key}' to be array-like, got {type(img_data)}")
                    img = torch.from_numpy(np.asarray(img_data).copy()).permute(2, 0, 1)  # (C, H, W)
                    img = img.float().div(255.0).unsqueeze(0)  # (1, C, H, W)
                else:
                    # Critical: No image data available - this is required for Pi05
                    raise ValueError(
                        f"No image data found in observation. Expected one of: "
                        f"'sensors::image_top::rgb' or keys from visual_input_features: {visual_features}. "
                        f"Available keys: {list(ob.keys())}"
                    )
            else:
                # No visual features defined - this is a configuration issue
                raise ValueError(
                    f"No visual input features defined in ArkMLContext and no default image key found. "
                    f"Pi05 requires visual input. Available observation keys: {list(ob.keys())}"
                )

        # Images: tensor, ensure [1, C, H, W] for all visual input features
        # Validate that visual_input_features is properly set
        visual_input_features = getattr(ArkMLContext, 'visual_input_features', [])
        if not visual_input_features:
            # If no visual features defined, just return with primary image
            return obs

        for cam_name in visual_input_features:
            # Try to get the specific camera data, fallback to primary image if not available
            cam_data = ob.get(cam_name)
            if cam_data is not None:
                if not isinstance(cam_data, (np.ndarray, list)):
                    raise ValueError(f"Expected visual input '{cam_name}' to be array-like, got {type(cam_data)}")
                cam_img = torch.from_numpy(np.asarray(cam_data).copy()).permute(2, 0, 1)  # (C, H, W)
                cam_img = cam_img.float().div(255.0).unsqueeze(0)  # (1, C, H, W)
                obs[cam_name] = cam_img
            else:
                # Use the primary image as fallback for missing camera data
                # This maintains tensor shape consistency across all cameras
                obs[cam_name] = img

        return obs

    def _callback_text_input(
        self, time_stamp: int, channel_name: str, msg: string_t
    ) -> None:
        """
        Service callback to read text prompt.
        Args:
            time_stamp: Callback time
            channel_name: Service channel id.
            msg: Message

        Returns:
            None
        """
        self.text_input = msg.data