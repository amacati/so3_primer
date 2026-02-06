from pathlib import Path

import fire
import gymnasium
import lsy_sim  #   noqa: F401
from lsy_rl.ddpg.policy import DDPGPolicy
from lsy_rl.wrappers.tensordict_wrapper import DefaultTensorDictWrapper

from rotations.modules.ddpg import Critic, GoalActor


def main(n_runs: int = 5, gui: bool = True):
    path = Path(__file__).parents[1] / "saves/pick_and_place_pose"
    latest = sorted(p for p in path.iterdir() if p.is_dir() and p.stem.startswith("2024"))[-1]
    print(f"Save path: {latest}")

    env = gymnasium.make_vec(
        "FR3PickAndPlacePose-v0",
        render_mode="human" if gui else None,
        action_type="tangent",
        obs_type="quat_plus",
        vectorization_mode="sync",
    )
    max_episode_steps = env.envs[0].spec.max_episode_steps
    env = DefaultTensorDictWrapper(env)

    policy = DDPGPolicy(
        actor=GoalActor(env.observation_space, env.action_space),
        critic=Critic(env.observation_space, env.action_space),
    )
    policy.load(latest / "policy.pt")

    sample = env.reset()
    success = 0
    for _ in range(n_runs * max_episode_steps):
        action = policy.action(sample["obs"])
        sample = env.step(action)
        sample["obs"] = sample["next_obs"]
        if sample["truncated"] and sample["reward"] == 0:
            success += 1
    print(f"Succes rate: {success / n_runs:.3f}")


if __name__ == "__main__":
    fire.Fire(main)
