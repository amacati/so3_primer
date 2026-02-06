import time
from typing import Literal

import fire
import gymnasium
import numpy as np

import rotations  # noqa: F401


def main(
    action: Literal["quat", "matrix", "r6", "euler", "tangent"],
    obs: Literal["quat", "matrix", "r6", "euler", "tangent"],
    reward_type: Literal["sparse", "dense"] = "sparse",
    control_mode: Literal["rel", "abs", "rel_scale"] = "rel",
    n_envs: int = 32,
    n_steps: int = 10_000,
    seed: int | None = None, 
    gui: bool = False
):
    """Test the Rotation environment using random actions.

    Args:
        action: The rotation type of the actions.
        obs: The rotation type of the observations.
        reward_type: The type of reward between dense or sparse.
        control_mode: The control mode of the environment. 
        n_envs: The number of environments to create.
        n_steps: The total number of environment steps to run.
        seed: The seed to use for the environment.
        gui: Flag for whether to render the environment or not.
    """
    env_kwargs = {
        "action_type": action, 
        "obs_type": obs,
        "reward_type": reward_type,
        "control_mode": control_mode
    }
    env = gymnasium.make_vec("Rotation-v0", num_envs=n_envs, **env_kwargs)
    env.reset(seed)
    
    ep_rets, tot_rets = np.zeros(n_envs), []
    for _ in range(0, n_steps, n_envs):
        action = env.action_space.sample()
        action = action / np.linalg.norm(action, axis=1, keepdims=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_rets += reward
        time.sleep(0.01)
        if np.any(np.logical_or(terminated, truncated)):
            print(f"Episodes undiscounted returns: \t {ep_rets.tolist()}")
            env.reset()
            tot_rets.append(ep_rets)
            ep_rets = np.zeros(n_envs)
        # TODO: Check why rendering doesn't work
        gui and env.render()
    tot_rets = np.array(tot_rets).flatten()
    print(f"Mean episode undiscounted return over {len(tot_rets)} episodes: \t {(tot_rets).mean():.2f}")

if __name__ == "__main__":
    fire.Fire(main)
