import torch


def l2_distance(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Returns the L2 distance between vector v1 and v2."""
    return torch.norm(v1 - v2)


def get_orientation_diff_in_radian(
    orn0: torch.Tensor, orn1: torch.Tensor
) -> torch.Tensor:
    """
    Returns the difference between two quaternion orientations in radians.

    Args:
        orn0 (torch.Tensor): (x, y, z, w) quaternion
        orn1 (torch.Tensor): (x, y, z, w) quaternion

    Returns:
        orn_diff (torch.Tensor): orientation difference in radians
    """
    # Compute the difference quaternion
    diff_quat = quat_distance(orn0, orn1)

    # Convert to axis-angle representation
    axis_angle = quat2axisangle(diff_quat)

    # The magnitude of the axis-angle vector is the rotation angle
    return torch.norm(axis_angle)


def quat_distance(quaternion1, quaternion0):
    """
    Returns distance between two quaternions, such that distance * quaternion0 = quaternion1
    Always returns the shorter rotation path.

    Args:
        quaternion1 (torch.tensor): (x,y,z,w) quaternion or (..., 4) batched quaternions
        quaternion0 (torch.tensor): (x,y,z,w) quaternion or (..., 4) batched quaternions

    Returns:
        torch.tensor: (x,y,z,w) quaternion distance or (..., 4) batched quaternion distances
    """
    # Compute dot product along the last axis (quaternion components)
    d = torch.sum(quaternion0 * quaternion1, dim=-1, keepdim=True)
    # If dot product is negative, negate one quaternion to get shorter path
    quaternion1 = torch.where(d < 0.0, -quaternion1, quaternion1)

    return quat_multiply(quaternion1, quat_inverse(quaternion0))


def quat_multiply(quaternion1: torch.Tensor, quaternion0: torch.Tensor) -> torch.Tensor:
    """
    Return multiplication of two quaternions (q1 * q0).

    Args:
        quaternion1 (torch.Tensor): (x,y,z,w) quaternion
        quaternion0 (torch.Tensor): (x,y,z,w) quaternion

    Returns:
        torch.Tensor: (x,y,z,w) multiplied quaternion
    """
    x0, y0, z0, w0 = quaternion0[0], quaternion0[1], quaternion0[2], quaternion0[3]
    x1, y1, z1, w1 = quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]

    return torch.stack(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ],
        dim=0,
    )


def quat_inverse(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Return inverse of quaternion.

    E.g.:
    >>> q0 = random_quaternion()
    >>> q1 = quat_inverse(q0)
    >>> torch.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
    True

    Args:
        quaternion (torch.tensor): (x,y,z,w) quaternion

    Returns:
        torch.tensor: (x,y,z,w) quaternion inverse
    """
    return quat_conjugate(quaternion) / torch.dot(quaternion, quaternion)


def quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Return conjugate of quaternion.

    Args:
        quaternion (torch.Tensor): (x,y,z,w) quaternion

    Returns:
        torch.Tensor: (x,y,z,w) quaternion conjugate
    """
    return torch.cat([-quaternion[:3], quaternion[3:]])


def random_quaternion(num_quaternions: int = 1) -> torch.Tensor:
    """
    Generate random rotation quaternions, uniformly distributed over SO(3).

    Arguments:
        num_quaternions (int): number of quaternions to generate (default: 1)

    Returns:
        torch.Tensor: A tensor of shape (num_quaternions, 4) containing random unit quaternions.
    """
    # Generate four random numbers between 0 and 1
    rand = torch.rand(num_quaternions, 4)

    # Use the formula from Ken Shoemake's "Uniform Random Rotations"
    r1 = torch.sqrt(1.0 - rand[:, 0])
    r2 = torch.sqrt(rand[:, 0])
    t1 = 2 * torch.pi * rand[:, 1]
    t2 = 2 * torch.pi * rand[:, 2]

    quaternions = torch.stack(
        [
            r1 * torch.sin(t1),
            r1 * torch.cos(t1),
            r2 * torch.sin(t2),
            r2 * torch.cos(t2),
        ],
        dim=1,
    )

    return quaternions


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.
    Args:
        quat (tensor): (..., 4) tensor where the final dim is (x,y,z,w) quaternion
    Returns:
        tensor: (..., 3) axis-angle exponential coordinates
    """
    # reshape quaternion
    quat_shape = quat.shape[:-1]  # ignore last dim
    quat = quat.reshape(-1, 4)
    # clip quaternion
    quat[:, 3] = torch.clip(quat[:, 3], -1.0, 1.0)
    # Calculate denominator
    den = torch.sqrt(1.0 - quat[:, 3] * quat[:, 3])
    # Map this into a mask

    # Create return array
    ret = torch.zeros_like(quat)[:, :3]
    idx = torch.nonzero(den).reshape(-1)
    ret[idx, :] = (quat[idx, :3] * 2.0 * torch.acos(quat[idx, 3]).unsqueeze(-1)) / den[
        idx
    ].unsqueeze(-1)

    # Reshape and return output
    ret = ret.reshape(
        list(quat_shape)
        + [
            3,
        ]
    )
    return ret


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Apply a quaternion rotation to a vector (equivalent to R.from_quat(x).apply(y))
    Args:
        quat (torch.Tensor): (4,) or (N, 4) or (N, 1, 4) quaternion in (x, y, z, w) format
        vec (torch.Tensor): (3,) or (M, 3) or (1, M, 3) vector to rotate
    Returns:
        torch.Tensor: (M, 3) or (N, M, 3) rotated vector
    """
    assert quat.shape[-1] == 4, "Quaternion must have 4 components in last dimension"
    assert vec.shape[-1] == 3, "Vector must have 3 components in last dimension"

    # Ensure quat is at least 2D and vec is at least 2D
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)

    # Ensure quat is (N, 1, 4) and vec is (1, M, 3)
    if quat.dim() == 2:
        quat = quat.unsqueeze(1)
    if vec.dim() == 2:
        vec = vec.unsqueeze(0)

    # Extract quaternion components
    qx, qy, qz, qw = quat.unbind(-1)

    # Compute the quaternion multiplication
    t = torch.stack(
        [
            2 * (qy * vec[..., 2] - qz * vec[..., 1]),
            2 * (qz * vec[..., 0] - qx * vec[..., 2]),
            2 * (qx * vec[..., 1] - qy * vec[..., 0]),
        ],
        dim=-1,
    )

    # Compute the final rotated vector
    result = vec + qw.unsqueeze(-1) * t + torch.cross(quat[..., :3], t, dim=-1)

    # Remove any extra dimensions
    return result.squeeze()
