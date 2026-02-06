import os
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation as R

from rotations.envs.actions import euler_scale


def process_chunk(seed: int, chunk_size: int, max_angle_pi: float) -> tuple[float, float]:
    """Process a chunk of samples and return max and mean angles."""
    rng = np.random.default_rng(seed)
    # Generate random rotations and actions for this chunk
    rotations = R.random(chunk_size, random_state=rng)
    actions = rng.uniform(-1, 1, size=(chunk_size, 3))
    # Apply actions and calculate angles
    euler_rots = R.from_euler(
        "xyz", rotations.as_euler("xyz") + euler_scale(actions, max_angle_pi, True)
    )
    angles = (rotations.inv() * euler_rots).magnitude() / np.pi
    return np.max(angles), np.mean(angles)


def main():
    n_samples = 100_000_000
    max_angle_pi = 0.2
    n_workers = os.cpu_count()
    chunk_size = n_samples // n_workers

    # Create pool of workers and distribute chunks
    with Pool(n_workers) as pool:
        results = pool.starmap(
            process_chunk, [(i, chunk_size, max_angle_pi) for i in range(n_workers)]
        )

    # Combine results from all workers
    max_angles, mean_angles = zip(*results)
    overall_max = max(max_angles)
    overall_mean = sum(mean_angles) / len(mean_angles)

    print(f"Maximum angle: {overall_max:.3f}π rad")
    print(f"Mean angle: {overall_mean:.3f}π rad")
    print(f"Difference from target: {(max_angle_pi - overall_max):.3f}π rad")


if __name__ == "__main__":
    main()
