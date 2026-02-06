import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from td3 import main

if __name__ == "__main__":
    N = 50
    wandb = True
    for action in ("tangent", "euler", "quat", "matrix"):
        for control_mode in ("rel", "abs"):
            for reward in ("dense", "sparse"):
                main(action=action, control_mode=control_mode, reward=reward, n_runs=N, wandb=wandb)
    for reward in ("dense", "sparse"):
        main(action="tangent", control_mode="rel_scale", reward=reward, n_runs=N, wandb=wandb)
