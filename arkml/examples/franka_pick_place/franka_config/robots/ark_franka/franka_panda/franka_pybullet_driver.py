from ark.system.pybullet.pybullet_robot_driver import BulletRobotDriver
from ark.tools.log import log


class FrankaPyBulletDriver(BulletRobotDriver):

    def __init__(
        self,
        component_name: str,
        component_config: dict[str, any] = None,
        client: bool = True,
    ) -> None:
        super().__init__(component_name, component_config, client)

    def pass_joint_group_control_cmd(
        self, control_mode: str, cmd: dict[str, float], **kwargs
    ) -> None:
        # duplicate the last joint command for the gripper as there is no mechanical binding in the URDF
        if "panda_finger_joint1" in cmd:
            panda_finger_joint1 = cmd.get("panda_finger_joint1")
            cmd["panda_finger_joint2"] = panda_finger_joint1
        super().pass_joint_group_control_cmd(control_mode, cmd, **kwargs)

    def get_ee_pose(self) -> dict[str, float]:
        """!Get the end-effector pose in the world frame."""
        end_effector_idx = 6
        position, orientation = self.client.getLinkState(
            self.ref_body_id, end_effector_idx
        )[:2]
        return {"position": position, "orientation": orientation}

    def pass_cartesian_control_cmd(
        self, control_mode: str, position, quaternion, **kwargs
    ) -> None:
        """!Send a Cartesian control command by computing inverse kinematics.

        @param control_mode One of ``position``, ``velocity`` or ``torque``.
        @param position List of 3 floats representing the desired XYZ position.
        @param quaternion List of 4 floats representing the desired orientation as a quaternion.
        @param kwargs Additional keyword arguments passed to joint control.
        @return ``None``
        """
        if not (len(position) == 3 and len(quaternion) == 4):
            raise ValueError(
                "Position must be 3 elements and quaternion must be 4 elements."
            )

        end_effector_idx = kwargs.get("end_effector_idx")

        # Compute IK solution
        try:
            joint_angles = self.client.calculateInverseKinematics(
                bodyUniqueId=self.ref_body_id,
                endEffectorLinkIndex=end_effector_idx,
                targetPosition=position,
                targetOrientation=quaternion,
            )
        except Exception as e:
            log.error(f"Inverse kinematics failed: {e}")
            return

        joint_angles = list(joint_angles)
        # add gripper control if specified
        if "gripper" in kwargs:
            gripper_position = kwargs["gripper"]
            if gripper_position is not None:
                # Assuming the gripper is controlled by the last two joints
                joint_angles[-2] = gripper_position
                joint_angles[-1] = (
                    gripper_position  # Assuming the gripper has two joints
                )

        self.client.setJointMotorControlArray(
            bodyUniqueId=self.ref_body_id,
            jointIndices=list(range(9)),  # Assuming 9 joints including gripper
            controlMode=self.client.POSITION_CONTROL,
            targetPositions=joint_angles,
        )
