import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from ppo import main

if __name__ == "__main__":
    N = 50
    wandb = True
    for action in ("tangent", "euler", "quat", "matrix"):
        for control_mode in ("rel", "abs"):
            main(action=action, control_mode=control_mode, n_runs=N, wandb=wandb)
    main(action="tangent", control_mode="rel_scale", n_runs=N, wandb=wandb)
