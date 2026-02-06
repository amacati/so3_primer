import logging
import os
from pathlib import Path
from typing import Any, Literal

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import fire
import gymnasium
import toml
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
from lsy_rl.core.transforms import to_transforms
from lsy_rl.sac import sac
from lsy_rl.sac.policy import SACPolicy
from lsy_rl.utils import load_config
from lsy_rl.wrappers.dict_to_tensordict import DictToTensorDict
from ml_collections import ConfigDict

import rotations  # noqa: F401
from rotations.logging import AngleSuccessCollector
from rotations.modules.sac import actors, critics
from rotations.modules.sac.activations import TanHMatrixOffset, TanHQuatOffset, TanHR6OrthoOffset

OFFSET_ACTS = {"matrix": TanHMatrixOffset, "quat": TanHQuatOffset, "r6": TanHR6OrthoOffset}


def train(
    config: ConfigDict,
    offset: bool = False,
    n_runs: int = 1,
    wandb_log: bool = False,
    seed: int | None = None,
    console_output: bool = True,
) -> list[dict[str, float]]:
    if wandb_log:
        wandb_api_key_path = Path(__file__).parents[2] / "secrets/wandb_api_key.secret"
        if not wandb_api_key_path.exists():
            raise FileNotFoundError(f"WandB API key not found at {wandb_api_key_path}")
        with open(wandb_api_key_path, "r") as f:
            wandb_api_key = f.read().rstrip("\n").lstrip("\n")
        wandb.login(key=wandb_api_key)

    action_type, obs_type = config.env.kwargs.action_type, config.env.kwargs.obs_type
    control_mode = config.env.kwargs.control_mode
    reward = config.env.kwargs.reward_type
    save_dir = (
        Path(__file__).parents[2] / "saves/sac" / action_type / obs_type / control_mode / reward
    )

    env = gymnasium.make_vec(config.env.name, num_envs=config.env.n_envs, **config.env.kwargs)
    eval_env = gymnasium.make_vec(config.env.name, num_envs=config.env.n_envs, **config.env.kwargs)
    env = DictToTensorDict(JaxToTorch(env, device=config.sac.device))
    eval_env = DictToTensorDict(JaxToTorch(eval_env, device=config.sac.device))

    config.sac = convert_transforms(config.sac)

    # Set config parameters
    if reward == "sparse":  # Use hindsight learning
        replay_buffer = HerVectorReplayBuffer(
            num_envs=env.num_envs,
            max_size=config.sac.buffer_size,
            reward_fn=env.unwrapped.compute_reward,
            device=config.sac.device,
        )
        config.sac.replay_buffer = replay_buffer

    results = []
    for i in range(n_runs):
        mem_logger = MemLogger(filter="eval/")
        logger = LoggerList([mem_logger])
        if wandb_log:
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                config=config.to_dict(),
                dir=save_dir,
                group=config.wandb.get("group"),
            )
            logger.append(WandBLogger())
        if console_output:
            logger.append(ConsoleLogger(filter="eval/"))

        config.sac.seed = seed if seed is None else seed + i * 100
        config.sac.checkpoint_path = (
            save_dir if seed is None else save_dir / f"seeds/s{config.sac.seed}"
        )
        config.sac.checkpoint_path.mkdir(parents=True, exist_ok=True)
        config.sac.logger = logger
        # Must be done here to create new actors and critics on every run
        actor = actors[action_type](env.observation_space, env.action_space)
        critic = critics[action_type](env.observation_space, env.action_space)
        # Replace output activation for "rel" with special action types (matrix, quat, r6)
        if offset:
            assert control_mode == "rel" and action_type in OFFSET_ACTS
            actor.squash_layer = OFFSET_ACTS[action_type]()
        config.sac.policy = SACPolicy(actor, critic)

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
        config.sac.eval_collector = eval_collector

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

        config.sac.rollout_collector = rollout_collector

        sac(train_envs=env, eval_envs=eval_env, **config.sac)
        # Save the config for reproducibility
        with open(save_dir / "cfg.toml", "w") as f:
            config_dict = config.to_dict()
            config_dict["sac"].pop("policy")
            config_dict["sac"].pop("action_tf")
            toml.dump(config_dict, f)
        results.append(mem_logger.data)
        if wandb_log:
            wandb.finish()
    env.close()
    eval_env.close()
    logger.stop()
    return results


def apply_overrides(config: ConfigDict, overrides: dict[str, Any]) -> ConfigDict:
    """Apply overrides to the config.

    Args:
        config: The config to override.
        overrides: The overrides to apply.

    Returns:
        The overridden config.
    """
    # Only override SAC specific parameters
    for key, value in overrides.items():
        assert key in config.sac, f"Key {key} not found in config.sac"
        config.sac[key] = value
    if hasattr(config, "overrides"):
        del config.overrides
    return config


def convert_transforms(config: ConfigDict) -> ConfigDict:
    """Convert the transforms to actual Transform objects."""
    if "action_tf" in config:
        config.action_tf = to_transforms(config.action_tf)
    if "train_action_tf" in config:
        config.train_action_tf = to_transforms(config.train_action_tf)
    if "eval_action_tf" in config:
        config.eval_action_tf = to_transforms(config.eval_action_tf)
    return config


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
    """Train the SAC agent on the given experiment.

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
    config = load_config(Path(__file__).parent / "config/sac.toml")
    assert f"{action}-matrix-{reward}" in config.overrides, "No overrides found"
    config = apply_overrides(config, config.overrides.get(f"{action}-matrix-{reward}", {}))
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
    train(config, offset, n_runs, wandb_log=wandb, seed=seed)


if __name__ == "__main__":
    logging.basicConfig()
    save_dir = Path(__file__).parents[2] / "saves"
    fire.Fire(main)
