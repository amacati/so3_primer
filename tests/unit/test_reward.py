from typing import Literal

import gymnasium
import numpy as np
import pytest
import torch
from gymnasium.wrappers.vector.jax_to_numpy import JaxToNumpy

import rotations  # noqa: F401
from rotations.rotations import RotType


def _test_return(
    x: np.ndarray | torch.Tensor,
    atype: type | None = None,
    shape: tuple[int, ...] | None = None,
    dtype: type | None = None,
    device: str | None = None,
):
    if atype is not None:
        assert isinstance(x, atype), f"Expected type {atype}, got {type(x)}"
    if shape is not None:
        assert x.shape == shape, f"Expected shape {shape}, got {x.shape}"
    if dtype is not None:
        assert x.dtype == dtype, f"Expected dtype {dtype}, got {x.dtype}"
    if device is not None:
        assert x.device == torch.device(device), f"Expected device {device}, got {x.device}"


@pytest.mark.unit
@pytest.mark.parametrize("action_type", RotType)
@pytest.mark.parametrize("obs_type", RotType)
@pytest.mark.parametrize("reward_type", ["sparse", "dense"])
def test_reward(action_type: RotType, obs_type: RotType, reward_type: Literal["sparse", "dense"]):
    batch = []
    obs_type = RotType(obs_type)
    env = gymnasium.make_vec(
        "Rotation-v0", action_type=action_type, obs_type=obs_type, reward_type=reward_type
    )
    env = JaxToNumpy(env)
    obs, _ = env.reset()
    batch.append(obs)
    for _ in range(10):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        batch.append(obs)
        assert obs in env.observation_space
    env.close()
    achieved_goal = np.array([obs["achieved_goal"] for obs in batch]).squeeze()
    desired_goal = np.array([obs["desired_goal"] for obs in batch]).squeeze()
    reward = env.unwrapped.compute_reward(achieved_goal, desired_goal)
    _test_return(reward, atype=np.ndarray, shape=(len(batch),))

    achieved_goal, desired_goal = torch.tensor(achieved_goal), torch.tensor(desired_goal)
    reward = env.unwrapped.compute_reward(achieved_goal, desired_goal)
    _test_return(reward, atype=torch.Tensor, shape=(len(batch),), dtype=torch.float32, device="cpu")

    if torch.cuda.is_available():
        reward = env.unwrapped.compute_reward(achieved_goal.cuda(), desired_goal.cuda())
        _test_return(
            reward, atype=torch.Tensor, shape=(len(batch),), dtype=torch.float32, device="cuda:0"
        )
