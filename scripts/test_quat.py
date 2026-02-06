from pathlib import Path

import fire
import gymnasium
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Dict
from lsy_rl.ddpg.policy import DDPGActorNetwork, DDPGCriticNetwork
from lsy_rl.utils import polyak_update_
from lsy_rl.wrappers.tensordict_wrapper import DefaultTensorDictWrapper
from tensordict import TensorDict
from torch import FloatTensor

import rotations  # noqa: F401


class QuatActor(nn.Module):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__()
        assert isinstance(obs_space, Dict), f"Invalid obs space type {type(obs_space)}"
        assert isinstance(action_space, Box), f"Invalid action space type {type(action_space)}"
        obs_dim = obs_space["observation"].shape[1] + obs_space["desired_goal"].shape[1]
        action_dim = action_space.shape[1]  # Remove num_envs dimension
        self.network = DDPGActorNetwork(obs_dim, action_dim)
        # Initialize the target network and synchronize the weights
        self.target_network = DDPGActorNetwork(obs_dim, action_dim)
        self.target_network.load_state_dict(self.network.state_dict())

    def forward(self, obs: TensorDict):
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        x = self.network(torch.cat([obs["observation"], obs["desired_goal"]], dim=-1))
        print(f"Norm: {x.norm(dim=-1, keepdim=True)}")
        return x / torch.norm(x, dim=-1, keepdim=True)

    def target(self, obs: TensorDict):
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        x = self.target_network(torch.cat([obs["observation"], obs["desired_goal"]], dim=-1))
        return x / torch.norm(x, dim=-1, keepdim=True)

    def update_target(self, tau: float):
        """Update the target network with the current weights."""
        polyak_update_(self.target_network, self.network, tau)


class Critic(nn.Module):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__()
        assert isinstance(obs_space, Dict), f"Invalid obs space type {type(obs_space)}"
        assert isinstance(action_space, Box), f"Invalid action space type {type(action_space)}"
        obs_dim = obs_space["observation"].shape[1] + obs_space["desired_goal"].shape[1]
        action_dim = action_space.shape[1]  # Remove num_envs dimension
        self.network = DDPGCriticNetwork(obs_dim + action_dim)
        # Initialize the target network and synchronize the weights
        self.target_network = DDPGCriticNetwork(obs_dim + action_dim)
        self.target_network.load_state_dict(self.network.state_dict())

    def forward(self, obs: TensorDict, action: FloatTensor):
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        return self.network(torch.cat([obs["observation"], obs["desired_goal"], action], dim=-1))

    def target(self, obs: TensorDict, action: FloatTensor):
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        return self.target_network(
            torch.cat([obs["observation"], obs["desired_goal"], action], dim=-1)
        )

    def update_target(self, tau: float):
        polyak_update_(self.target_network, self.network, tau)


def main(n_tests: int = 1, gui: bool = True):
    env = gymnasium.make_vec(
        "RotationQuat-v0", vectorization_mode="sync", render_mode="human" if gui else None
    )
    env = DefaultTensorDictWrapper(env)
    path = Path(__file__).parents[1] / "saves/quat"
    save_path = sorted(p for p in path.glob("*") if p.name[0].isdigit())[-1] / "policy.pt"
    actor = QuatActor(env.observation_space, env.action_space)
    actor.load_state_dict(torch.load(save_path)["actor"])

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
