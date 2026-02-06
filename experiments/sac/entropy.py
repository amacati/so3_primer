import copy
import os
from pathlib import Path
from typing import Literal

import fire
from lsy_rl.utils import load_config
from ml_collections import ConfigDict
from sac import apply_overrides, train

from rotations.rotations import RotType


def train_with_entropy(
    config: ConfigDict,
    save_dir: Path,
    n_runs: int,
    wandb: bool,
    seed: int | None = None,
) -> None:
    """Train function wrapper for the sweep over target entropies.

    Args:
        config: The configuration dictionary.
        save_dir: The root directory for storing all runs.
        n_runs: The number of runs to perform.
        wandb: Whether to use Weights and Biases for logging.
        seed: Optional random seed.
    """
    # Train the agent using SAC
    print(f"Starting training for agent with scale factor = {config.scale_factor}\n\n")
    train(config, n_runs=n_runs, wandb_log=wandb, seed=seed, console_output=True)

    # Move the stored checkpoint from the root directory to a subdirectory
    subdir = save_dir / "entropy" / f"sf_{config.scale_factor:.2f}"
    subdir.mkdir(parents=True, exist_ok=True)
    for child in save_dir.iterdir():
        if child.is_file() or child.name == "wandb" or child.name == "seeds":
            child.rename(subdir / child.name)


def main(
    action: Literal["quat", "matrix", "r6", "euler", "tangent"],
    obs: Literal["quat", "matrix", "r6", "euler", "tangent"],
    control_mode: Literal["rel", "abs", "rel_scale"] = "rel",
    reward: Literal["dense", "sparse"] = "sparse",
    n_runs: int = 1,
    scale_factors: list[float] = [1.0],
    wandb: bool = False,
    seed: int | None = None,
    group: str | None = None,
):
    """Perform a sweep over scale factors for the target entropy of SAC using the
    given environment parameters.

    Args:
        action: The action type to use.
        obs: The observation type to use.
        control_mode: The control mode to use.
        reward: The type of reward signal to use.
        n_runs: The number of runs to perform per scale factor.
        scale_factors: Scale factors for the target entropy.
        wandb: Whether to use Weights and Biases for logging.
        seed: The seed to use for the experiment.
        group: Overwrite the WandB group name.
    """
    # Load the base config
    config = load_config(Path(__file__).parent / "config/sac.toml")
    config = apply_overrides(
        config, config.overrides.get(f"{action}-{obs}-{reward}", {})
    )
    config.env.kwargs["action_type"] = action
    config.env.kwargs["obs_type"] = obs
    config.env.kwargs["control_mode"] = control_mode
    config.env.kwargs["reward_type"] = reward
    save_dir = (
        Path(__file__).parents[2] / "saves/sac" / action / obs / control_mode / reward
    )

    # Set WandB's config parameters
    config.wandb.project = "rot|sac_entropy"
    config.wandb.entity = "lsy-tum"

    # Sweep over entropy target scale factors
    action_dim = RotType(action).dim
    for scale_factor in scale_factors:
        config_copy = copy.deepcopy(config)
        config_copy.scale_factor = scale_factor
        config_copy.sac.target_entropy = -action_dim * scale_factor
        if group is not None:
            config_copy.wandb.group = group
        else:
            config_copy.wandb.group = (
                f"{action}|{obs}|{control_mode}|{reward}|sf_{scale_factor:.2f}"
            )
        train_with_entropy(config_copy, save_dir, n_runs, wandb, seed)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    fire.Fire(main)
