from pathlib import Path

import fire
import gymnasium
import lsy_sim  #   noqa: F401
import torch
from lsy_rl.ddpg.policy import DDPGPolicy
from lsy_rl.wrappers.tensordict_wrapper import DefaultTensorDictWrapper
from tqdm import tqdm

from rotations.modules.ddpg import Critic, GoalActor


@torch.no_grad
def main(
    n_runs: int = 5,
    id_save: int = -1,
    action_type: str = "tangent",
    obs_type: str = "quat_plus",
    gui: bool = True,
):
    path = Path(__file__).parents[1] / "saves/fr3_reach_rot"
    latest = sorted(p for p in path.iterdir() if p.is_dir() and p.stem.startswith("2024"))[id_save]
    print(latest)

    env = gymnasium.make_vec(
        "FR3ReachRot-v0",
        render_mode="human" if gui else None,
        action_type=action_type,
        obs_type=obs_type,
        vectorization_mode="sync",
    )
    env = DefaultTensorDictWrapper(env)

    policy = DDPGPolicy(
        actor=GoalActor(env.observation_space, env.action_space),
        critic=Critic(env.observation_space, env.action_space),
    )
    policy.load(latest / "policy.pt")

    sample = env.reset()
    i, success = 0, 0
    pbar = tqdm(total=n_runs)
    while i < n_runs:
        action = policy.action(sample["obs"])
        sample = env.step(action)
        sample["obs"] = sample["next_obs"]
        done = sample["terminated"] | sample["truncated"]
        i += done.sum()
        success += (sample["reward"][done] == 0).sum()
        pbar.update(done.sum().item())
    print(f"Success rate: {100 * success / i:.1f}% ({i} runs)")


if __name__ == "__main__":
    fire.Fire(main)
