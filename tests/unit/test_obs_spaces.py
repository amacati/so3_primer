import gymnasium
import pytest
from gymnasium.wrappers.vector.jax_to_numpy import JaxToNumpy

import rotations  # noqa: F401
from rotations.rotations import RotType


@pytest.mark.unit
@pytest.mark.parametrize("action_type", RotType)
@pytest.mark.parametrize("obs_type", RotType)
def test_vec_obs_space(action_type: RotType, obs_type: RotType):
    env = JaxToNumpy(gymnasium.make_vec("Rotation-v0", action_type=action_type, obs_type=obs_type))
    obs, _ = env.reset()
    if not (obs_shape := env.observation_space["observation"].shape) == (
        env.num_envs,
        obs_type.dim,
    ):
        raise AssertionError(f"Shape {obs_shape} != ({env.num_envs}, {obs_type.dim}) ({obs_type})")
    assert obs in env.observation_space
    for _ in range(10):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs in env.observation_space
    env.close()
