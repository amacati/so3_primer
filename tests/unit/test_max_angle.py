import gymnasium
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import rotations  # noqa: F401
from rotations.rotations import RotType

max_angle_actions = {
    RotType.euler: np.array([[1.0, 1.0, 1.0]]),
    RotType.quat: np.array([[1.0, 0.0, 0.0, 0.0]]),
    RotType.quat_plus: np.array([[1.0, 0.0, 0.0, 0.0]]),
    RotType.tangent: np.array([[1.0, 1.0, 1.0]]),
    RotType.matrix: np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    RotType.r6: np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]]),
}


def _test_max_rotation(
    env: gymnasium.Env,
    action: np.ndarray,
    max_angle: float,
    equal: bool = True,
    ref: np.ndarray | None = None,
):
    """Test that the maximum rotation with the given action is approximately the max_angle.

    Args:
        env: The environment to test.
        action: The action to apply.
        max_angle: The maximum angle in fractions of pi.
        equal: Whether the angle should be equal to the max_angle or less than it.
        ref: The reference quaternion to compare the rotation to.
    """
    env.reset()
    ref = R.identity() if ref is None else ref
    env.unwrapped.data = env.unwrapped.data.replace(q=ref.as_quat().reshape(1, 4))
    env.step(action)
    rot = R.from_quat(env.unwrapped.data.q)
    angle = (rot.inv() * ref).magnitude()[0] / np.pi
    if not equal:
        assert angle - max_angle < 1e-6, f"Angle {angle:.4f}π not less than {max_angle}π"
        return
    assert np.abs(angle - max_angle) < 1e-6, f"Angle {angle:.4f}π not equal to {max_angle}π"


@pytest.mark.unit
@pytest.mark.parametrize("step_len", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("action_type", RotType)
@pytest.mark.parametrize("control_mode", ["rel", "rel_scale", "abs"])
def test_max_rotation(action_type: RotType, step_len: float, control_mode: str):
    """Test that the maximum rotation with Euler angle actions is pi/10."""
    env = gymnasium.make_vec(
        "Rotation-v0", action_type=action_type, step_len=step_len, control_mode=control_mode
    )
    action = max_angle_actions[action_type]
    equal = (control_mode != "rel_scale") or action_type != RotType.euler
    _test_max_rotation(env, action, step_len, equal=equal)
    for _ in range(100):
        action = env.action_space.sample()
        _test_max_rotation(env, action, step_len, equal=False, ref=R.random())
