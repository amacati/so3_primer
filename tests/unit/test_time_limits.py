import gymnasium
import pytest

import rotations  # noqa: F401
from rotations.rotations import RotType


@pytest.mark.unit
@pytest.mark.parametrize("max_steps", [25, 50, 100])
@pytest.mark.parametrize("action_type", RotType)
@pytest.mark.parametrize("obs_type", RotType)
def test_time_limits(max_steps: int, action_type: RotType, obs_type: RotType):
    """Test that the environment respects the time limits."""
    env = gymnasium.make_vec(
        "Rotation-v0", action_type=action_type, obs_type=obs_type, max_steps=max_steps
    )
    env.reset()
    steps = 0
    terminated, truncated = False, False
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            break
    assert steps == max_steps, f"Steps {steps} not equal to {max_steps}"
