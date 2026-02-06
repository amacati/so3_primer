import copy
import os
from decimal import Decimal
from pathlib import Path
from typing import Literal

import fire
import numpy as np
from lsy_rl.utils import load_config
from ml_collections import ConfigDict
from ppo import apply_overrides, train


def train_with_entropy(
    config: ConfigDict, save_dir: Path, n_runs: int, wandb: bool, seed: int | None = None
) -> None:
    """Train function wrapper for the sweep over entropy coefficients.

    Args:
        config: The configuration dictionary.
        save_dir: The root directory for storing all runs.
        n_runs: The number of runs to perform.
        wandb: Whether to use Weights and Biases for logging.
        seed: Optional random seed.
    """
    # Train the agent using PPO
    print(f"Starting training for agent with entropy coefficient = {config.ppo.ent_coef}\n\n")
    train(config, n_runs=n_runs, wandb_log=wandb, seed=seed, console_output=True)

    # Move the stored checkpoint from the root directory to a subdirectory
    subdir = (
        save_dir / "entropy" / f"ent_{Decimal(config.ppo.ent_coef):.3E}"
        if not config.ppo.use_logstd_net
        else save_dir / "logstd" / "entropy" / f"ent_{Decimal(config.ppo.ent_coef):.3E}"
    )
    subdir.mkdir(parents=True, exist_ok=True)
    for child in save_dir.iterdir():
        if child.is_file() or child.name == "wandb" or child.name == "seeds":
            child.rename(subdir / child.name)


def main(
    action: Literal["quat", "matrix", "r6", "euler", "tangent"],
    obs: Literal["quat", "matrix", "r6", "euler", "tangent"],
    control_mode: Literal["rel", "abs", "rel_scale"] = "rel",
    reward: Literal["dense", "sparse"] = "dense",
    n_coeffs: int = 10,
    n_runs: int = 1,
    ent_coeff: float | list[float] = 0.5,
    log_scale: bool = False,
    wandb: bool = False,
    seed: int | None = None,
    group: str | None = None,
):
    """Perform a sweep over entropy coefficients for the PPO agent using the
    given environment parameters.

    Args:
        action: The action type to use.
        obs: The observation type to use.
        control_mode: The control mode to use.
        reward: The type of reward signal to use.
        n_coeffs: The number of entropy coefficients to test.
        n_runs: The number of runs to perform per coefficient.
        ent_coeff: Entropy coefficients to use if list or max value if a single number is passed.
        log_scale: Whether to use a log or linear scale for sampling intermediate entropy coefficients.
        wandb: Whether to use Weights and Biases for logging.
        seed: The seed to use for the experiment.
        group: Overwrite the WandB group name.
    """
    # Load the base config
    config = load_config(Path(__file__).parent / "config/ppo.toml")
    config = apply_overrides(config, config.overrides.get(f"{action}-{obs}", {}))
    config.env.kwargs["action_type"] = action
    config.env.kwargs["obs_type"] = obs
    config.env.kwargs["control_mode"] = control_mode
    config.env.kwargs["reward_type"] = reward
    save_dir = Path(__file__).parents[2] / "saves/ppo" / action / obs / control_mode

    # Determine entropy coefficients
    if isinstance(ent_coeff, list):
        assert all([c >= 0 for c in ent_coeff]), (
            f"Entropy coefficients must be positive, is {ent_coeff}"
        )
        ent_coeffs = ent_coeff
    else:
        ent_coeffs = (
            np.geomspace(1e-8, ent_coeff, n_coeffs)
            if log_scale
            else np.linspace(0, ent_coeff, n_coeffs)
        )

    # Set WandB's config parameters
    config.wandb.project = "rot|ppo_entropy"
    config.wandb.entity = "lsy-tum"
    if group is not None:
        config.wandb.group = group
    else:
        config.wandb.group = f"{action}|{obs}|{control_mode}|ent_{Decimal(ent_coeffs[-1]):.3E}"

    # Sweep over entropy coefficients
    for ent_coeff in ent_coeffs:
        config_copy = copy.deepcopy(config)
        config_copy.ppo.ent_coef = ent_coeff
        train_with_entropy(config_copy, save_dir, n_runs, wandb, seed)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    fire.Fire(main)
