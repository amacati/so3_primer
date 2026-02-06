import gymnasium
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from rotations.rotations import RotType


def rel_control(action_type: RotType, r_is: R, r_goal: R) -> R:
    if action_type is RotType.euler:
        return (r_goal.as_euler("xyz") - r_is.as_euler("xyz")) / np.array([np.pi, np.pi / 2, np.pi])
    return action_type.as_array(r_is.inv() * r_goal)


def abs_control(action_type: RotType, r_is: R, r_goal: R) -> R:
    return action_type.as_array(r_goal)


@pytest.mark.parametrize("action_type", RotType)
@pytest.mark.parametrize("control_mode", ["rel", "abs"])
def test_rotation_solutions(action_type: RotType, control_mode: str):
    env = gymnasium.make_vec(
        "Rotation-v0", num_envs=1, action_type=action_type, control_mode=control_mode
    )
    obs, _ = env.reset(seed=42)

    ctrl = rel_control if control_mode == "rel" else abs_control

    while True:
        r_is, r_goal = R.from_quat(obs["observation"]), R.from_quat(obs["desired_goal"])
        # Scaling if desired:
        # step_len = env.unwrapped.data.step_len
        # drot = drot ** min(1 / step_len, np.pi / (drot.magnitude()[0] + 1e-5))
        action = ctrl(action_type, r_is, r_goal)
        assert (-1 <= action).all() and (action <= 1).all(), "Action out of bounds"
        obs, reward, _, truncated, _ = env.step(action)
        if truncated.all():
            break
    mean_solved = (reward == 0).mean()
    assert mean_solved == 1.0, f"Could not solve the environment ({mean_solved:.2%})"
