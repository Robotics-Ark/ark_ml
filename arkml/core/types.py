from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np

Array = np.ndarray

@dataclass
class Observation:
    data: Dict[str, Array]     # e.g., {"cube": (3,), "target": (3,), "gripper": (1,), "franka_ee": (7,)}

@dataclass
class Action:
    data: Dict[str, Array]     # e.g., {"position": (3,), "quaternion": (4,), "gripper": (1,)}

@dataclass
class TimeStep:
    obs: Observation
    action: Optional[Action]
    reward: Optional[float] = None
    done: Optional[bool] = None
    info: Optional[Dict[str, Any]] = None

@dataclass
class Trajectory:
    steps: Tuple[TimeStep, ...]