from pathlib import Path

import fire
import gymnasium
import torch
from lsy_rl.wrappers.tensordict_wrapper import DefaultTensorDictWrapper

import rotations  # noqa: F401
from rotations.modules.ddpg import EulerActor


def main(n_tests: int = 1, gui: bool = True):
    env = gymnasium.make_vec(
        "RotationEuler-v0",
        vectorization_mode="sync",
        render_mode="human" if gui else None,
        obs_type="euler",
    )
    env = DefaultTensorDictWrapper(env)
    path = Path(__file__).parents[1] / "saves/euler"
    save_path = sorted(p for p in path.glob("*") if p.name[0].isdigit())[-1] / "policy.pt"
    actor = EulerActor(env.observation_space, env.action_space)
    actor.load_state_dict(torch.load(save_path, weights_only=True)["actor"])

    success = 0
    with torch.no_grad():
        for _ in range(n_tests):
            sample = env.reset()
            done = False
            while not done:
                action = actor(sample["obs"])
                sample = env.step(action)
                gui and env.unwrapped.envs[0].render()
                done = sample["terminated"] or sample["truncated"]
                sample["obs"] = sample["next_obs"]
            if sample["reward"] == 0:
                success += 1
        print(f"Success rate: {success / n_tests}")


if __name__ == "__main__":
    fire.Fire(main)
