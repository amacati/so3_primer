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
from lsy_rl.core.transforms import to_transforms
from lsy_rl.ppo import ppo
from lsy_rl.ppo.policy import PPOPolicy
from lsy_rl.utils import load_config
from ml_collections import ConfigDict

import rotations  # noqa: F401
from rotations.logging import AngleSuccessCollector
from rotations.modules.ddpg.activations import TanHMatrixOffset, TanHQuatOffset, TanHR6OrthoOffset
from rotations.modules.ppo import actors, critics
from rotations.wrappers.flatten import FlattenObs

OFFSET_ACTS = {"matrix": TanHMatrixOffset, "quat": TanHQuatOffset, "r6": TanHR6OrthoOffset}


def train(
    config: ConfigDict,
    offset: bool = False,
    n_runs: int = 1,
    wandb_log: bool = False,
    seed: int | None = None,
    console_output: bool = True,
) -> list[dict[str, float]]:
    assert isinstance(seed, int | None), f"Seed should be int or None, is {type(seed)}"
    if wandb_log:
        wandb_api_key_path = Path(__file__).parents[2] / "secrets/wandb_api_key.secret"
        if not wandb_api_key_path.exists():
            raise FileNotFoundError(f"WandB API key not found at {wandb_api_key_path}")
        with open(wandb_api_key_path, "r") as f:
            wandb_api_key = f.read().rstrip("\n").lstrip("\n")
        wandb.login(key=wandb_api_key)

    action_type, obs_type = config.env.kwargs.action_type, config.env.kwargs.obs_type
    control_mode = config.env.kwargs.control_mode
    save_dir = Path(__file__).parents[2] / "saves/ppo" / action_type / obs_type / control_mode
    wandb_config = config.to_dict()  # Save before converting TFs

    config.ppo = convert_tfs(config.ppo)
    config.env.kwargs.reward_type = "dense"  # PPO only supports dense rewards
    env = gymnasium.make_vec(config.env.name, num_envs=config.env.n_envs, **config.env.kwargs)
    eval_env = gymnasium.make_vec(config.env.name, num_envs=config.env.n_envs, **config.env.kwargs)
    env = FlattenObs(JaxToTorch(env, device=config.ppo.device))
    eval_env = FlattenObs(JaxToTorch(eval_env, device=config.ppo.device))

    results = []
    for i in range(n_runs):
        mem_logger = MemLogger()
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

        config.ppo.seed = seed if seed is None else seed + i * 100
        config.ppo.checkpoint_path = (
            save_dir if seed is None else save_dir / f"seeds/s{config.ppo.seed}"
        )
        config.ppo.checkpoint_path.mkdir(parents=True, exist_ok=True)

        action_shape = env.single_action_space.shape
        obs_shape = env.single_observation_space.shape
        use_logstd_net = config.ppo.use_logstd_net
        actor = actors[action_type](obs_shape, action_shape, use_logstd_net=use_logstd_net)
        # Improved log-std initialization because default of 0.0 is too high for rotation tasks
        actor.network.logstd.data.fill_(-2.0)
        if action_type == "tangent" and control_mode == "rel_scale":
            # Initialize the scale to 0.0 for rel_scale tangent
            actor.network.logstd.data.fill_(0.0)
        critic = critics[action_type](obs_shape)
        # Replace output activation for "rel" with special action types (matrix, quat, r6)
        if offset:
            assert control_mode == "rel" and action_type in OFFSET_ACTS
            actor.network.network["f_out"] = OFFSET_ACTS[action_type]()
        policy = PPOPolicy(actor, critic)

        rollout_log_collector = CollectorList()
        rollout_log_collector.append(
            LogCollector(target="reward", log_key="rollout/reward", reduce="sum")
        )
        rollout_log_collector.append(
            LogCollector(target="reward", log_key="rollout/steps", reduce="cnt")
        )
        rollout_log_collector.append(
            AngleSuccessCollector(
                angle_key="rollout/angle", success_key="rollout/success", tol=env.unwrapped.data.tol
            )
        )
        eval_log_collector = CollectorList()
        eval_log_collector.append(
            LogCollector(target="reward", log_key="eval/reward", reduce="sum")
        )
        eval_log_collector.append(LogCollector(target="reward", log_key="eval/steps", reduce="cnt"))
        eval_log_collector.append(
            AngleSuccessCollector(
                angle_key="eval/angle", success_key="eval/success", tol=env.unwrapped.data.tol
            )
        )

        config.ppo.rollout_log_collector = rollout_log_collector
        config.ppo.eval_log_collector = eval_log_collector
        policy = ppo(train_envs=env, eval_envs=eval_env, **config.ppo, logger=logger, agent=policy)
        # Save the config for reproducibility
        with open(save_dir / "cfg.toml", "w") as f:
            toml.dump(config.to_dict(), f)
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
    # Only override PPO specific parameters
    for key, value in overrides.items():
        config.ppo[key] = value
    return config


def convert_tfs(config: ConfigDict) -> ConfigDict:
    """Convert the action transforms to the correct format."""
    for key, value in config.items():
        if key == "action_tf":
            config.action_tf = to_transforms(value).to(config.device)
    return config


def main(
    action: Literal["quat", "matrix", "r6", "euler", "tangent"],
    control_mode: Literal["rel", "abs", "rel_scale"] = "rel",
    offset: bool = False,
    n_runs: int = 1,
    wandb: bool = False,
    seed: int | None = None,
    group: str | None = None,
):
    """Train the PPO agent on the given experiment.

    Args:
        action: Rotation type to use for actions.
        control_mode: Control mode for orientations.
        offset: Add identity offset to non-zero centered rel actions spaces (quat, matrix, r6).
        n_runs: The number of runs to perform.
        wandb: Whether to use Weights and Biases for logging.
        seed: The seed to use for the experiment.
        group: Overwrite the WandB group name.
    """
    config = load_config(Path(__file__).parent / "config/ppo.toml")
    config = apply_overrides(config, config.overrides.get(f"{action}-matrix", {}))
    config.env.kwargs["action_type"] = action
    config.env.kwargs["obs_type"] = "matrix"
    config.env.kwargs["control_mode"] = control_mode
    if group is not None:
        config.wandb.group = group
    else:
        config.wandb.group = f"{action}|{control_mode}"
        if offset:
            config.wandb.group += "|offset"
    train(config, offset, n_runs, wandb_log=wandb, seed=seed)


if __name__ == "__main__":
    logging.basicConfig()
    save_dir = Path(__file__).parents[2] / "saves"
    fire.Fire(main)
