import logging
import os
from pathlib import Path
from typing import Any, Literal

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import fire
import gymnasium
import toml
import torch.nn as nn
import wandb
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from lsy_rl.core.logger import (
    CollectorList,
    ConsoleLogger,
    LogCollector,
    LoggerList,
    MemLogger,
    WandBLogger,
)
from lsy_rl.core.replay_buffer import HerVectorReplayBuffer
from lsy_rl.core.transforms import FunctionalTF, to_transforms
from lsy_rl.td3 import td3
from lsy_rl.td3.policy import TD3Policy
from lsy_rl.utils import load_config
from lsy_rl.wrappers.dict_to_tensordict import DictToTensorDict
from ml_collections import ConfigDict

import rotations  # noqa: F401
from rotations.logging import AngleSuccessCollector
from rotations.modules.ddpg.activations import TanHMatrixOffset, TanHQuatOffset, TanHR6OrthoOffset
from rotations.modules.td3 import actors, critics
from rotations.projections import action_projections

OFFSET_ACTS = {"matrix": TanHMatrixOffset, "quat": TanHQuatOffset, "r6": TanHR6OrthoOffset}


def apply_overrides(config: ConfigDict, overrides: dict[str, Any]) -> ConfigDict:
    """Apply overrides to the config.

    Args:
        config: The config to override.
        overrides: The overrides to apply.

    Returns:
        The overridden config.
    """
    # Only override TD3 specific parameters
    for key, value in overrides.items():
        assert key in config.td3, f"Key {key} not found in config.td3"
        config.td3[key] = value
    if hasattr(config, "overrides"):
        del config.overrides
    return config


def convert_transforms(config: ConfigDict) -> ConfigDict:
    """Convert the transforms to actual Transform objects."""
    if "action_tf" in config:
        config.action_tf = to_transforms(config.action_tf)
    if "target_action_tf" in config:
        config.target_action_tf = to_transforms(config.target_action_tf)
    if "train_action_tf" in config:
        config.train_action_tf = to_transforms(config.train_action_tf)
    if "eval_action_tf" in config:
        config.eval_action_tf = to_transforms(config.eval_action_tf)
    return config


def train(
    config: ConfigDict,
    offset: bool = False,
    n_runs: int = 1,
    actor_cls: nn.Module | None = None,
    critic_cls: nn.Module | None = None,
    wandb_log: bool = False,
    seed: int | None = None,
    console_output: bool = True,
) -> list[dict[str, float]]:
    if wandb_log:
        wandb_api_key_path = Path(__file__).parents[2] / "secrets/wandb_api_key.secret"
        if not wandb_api_key_path.exists():
            raise FileNotFoundError(f"WandB API key not found at {wandb_api_key_path}")
        with open(Path(__file__).parents[2] / "secrets/wandb_api_key.secret", "r") as f:
            wandb_api_key = f.read().rstrip("\n").lstrip("\n")
        wandb.login(key=wandb_api_key)

    action_type, obs_type = config.env.kwargs.action_type, config.env.kwargs.obs_type
    control_mode = config.env.kwargs.control_mode
    reward = config.env.kwargs.reward_type
    save_dir = (
        Path(__file__).parents[2] / "saves/td3" / action_type / obs_type / control_mode / reward
    )
    wandb_config = config.to_dict()

    env = gymnasium.make_vec(config.env.name, num_envs=config.env.n_envs, **config.env.kwargs)
    eval_env = gymnasium.make_vec(config.env.name, num_envs=config.env.n_envs, **config.env.kwargs)
    env = DictToTensorDict(JaxToTorch(env, device=config.td3.device), device=config.td3.device)
    eval_env = DictToTensorDict(
        JaxToTorch(eval_env, device=config.td3.device), device=config.td3.device
    )

    config.td3 = convert_transforms(config.td3)
    if "action_tf" in config.td3:
        config.td3.action_tf.append(FunctionalTF(action_projections[action_type]))
    if "target_action_tf" in config.td3:
        config.td3.target_action_tf.append(FunctionalTF(action_projections[action_type]))

    # Set config parameters
    if reward == "sparse":  # Use hindsight learning
        replay_buffer = HerVectorReplayBuffer(
            num_envs=env.num_envs,
            max_size=config.td3.buffer_size,
            reward_fn=env.unwrapped.compute_reward,
            device=config.td3.device,
        )
        config.td3.replay_buffer = replay_buffer

    results = []
    for i in range(n_runs):
        mem_logger = MemLogger(filter="eval/")
        logger = LoggerList([mem_logger])
        if wandb_log:
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                config=wandb_config,
                dir=save_dir,
                group=config.wandb.get("group"),
            )
            logger.append(WandBLogger())
        if console_output:
            logger.append(ConsoleLogger(filter="eval/"))

        config.td3.seed = seed if seed is None else seed + i * 100
        config.td3.checkpoint_path = (
            save_dir if seed is None else save_dir / f"seeds/s{config.td3.seed}"
        )
        config.td3.checkpoint_path.mkdir(parents=True, exist_ok=True)
        config.td3.logger = logger
        # Must be done here to create new actors and critics on every run
        if actor_cls is not None:
            assert critic_cls is not None, "Critic class must be set if actor class is not None"
            actor = actor_cls(env.observation_space, env.action_space)
            critic = critic_cls(env.observation_space, env.action_space)
            # Replace output activation for "rel" with special action types (matrix, quat, r6)
            if offset:
                assert control_mode == "rel" and action_type in OFFSET_ACTS
                actor.network.network["f_output"] = OFFSET_ACTS[action_type]()
                actor.target_network.network["f_output"] = OFFSET_ACTS[action_type]()
            config.td3.policy = TD3Policy(actor, critic)

        if reward == "sparse":  # Only then is the buffer defined
            replay_buffer.clear()  # Ensure we do not leak experience from previous runs

        # Create collectors for logging
        eval_collector = CollectorList()
        eval_collector.append(LogCollector(target="reward", log_key="eval/reward", reduce="sum"))
        eval_collector.append(LogCollector(target="reward", log_key="eval/step", reduce="cnt"))
        eval_collector.append(
            AngleSuccessCollector(
                angle_key="eval/angle", success_key="eval/success", tol=env.unwrapped.data.tol
            )
        )
        config.td3.eval_collector = eval_collector

        rollout_collector = CollectorList()
        rollout_collector.append(
            LogCollector(target="reward", log_key="rollout/reward", reduce="sum")
        )
        rollout_collector.append(
            LogCollector(target="reward", log_key="rollout/step", reduce="cnt")
        )
        rollout_collector.append(
            AngleSuccessCollector(
                angle_key="rollout/angle", success_key="rollout/success", tol=env.unwrapped.data.tol
            )
        )

        config.td3.rollout_collector = rollout_collector

        td3(env, eval_env, **config.td3)
        # Save the config for reproducibility
        with open(save_dir / "cfg.toml", "w") as f:
            config_dict = config.to_dict()
            config_dict["td3"].pop("policy")
            config_dict["td3"].pop("action_tf")
            config_dict["td3"].pop("target_action_tf")
            toml.dump(config_dict, f)
        results.append(mem_logger.data)
        if wandb_log:
            wandb.finish()
    env.close()
    eval_env.close()
    logger.stop()
    return results


def main(
    action: Literal["quat", "matrix", "r6", "euler", "tangent"],
    control_mode: Literal["rel", "abs", "rel_scale"] = "rel",
    reward: Literal["dense", "sparse"] = "sparse",
    offset: bool = False,
    n_runs: int = 1,
    wandb: bool = False,
    seed: int | None = None,
    group: str | None = None,
):
    """Train the TD3 agent on the given experiment.

    Args:
        action: Rotation type to use for actions.
        control_mode: Control mode for orientations.
        reward: Reward type to use for the environment.
        offset: Add identity offset to non-zero centered rel actions spaces (quat, matrix, r6).
        n_runs: The number of runs to perform.
        wandb: Whether to use Weights and Biases for logging.
        seed: The seed to use for the experiment.
        group: Overwrite the WandB group name.
    """
    config = load_config(Path(__file__).parent / "config/td3.toml")
    assert f"{action}-matrix-{reward}" in config.overrides, "No overrides found"
    config = apply_overrides(
        config, config.get("overrides", {}).get(f"{action}-matrix-{reward}", {})
    )
    config.env.kwargs["action_type"] = action
    config.env.kwargs["obs_type"] = "matrix"
    config.env.kwargs["control_mode"] = control_mode
    config.env.kwargs["reward_type"] = reward
    if group is not None:
        config.wandb.group = group
    else:
        config.wandb.group = f"{action}|{control_mode}|{reward}"
        if offset:
            config.wandb.group += "|offset"
    train(
        config,
        offset,
        n_runs,
        actor_cls=actors[action],
        critic_cls=critics[action],
        wandb_log=wandb,
        seed=seed,
    )


if __name__ == "__main__":
    logging.basicConfig()
    fire.Fire(main)
