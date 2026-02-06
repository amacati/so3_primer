from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.spatial.transform import Rotation as JR

from rotations.envs.actions import rel_euler_rotation


@partial(jax.jit, static_argnames=["n_samples"])
def sample_random_rotations(key: jr.PRNGKey, n_samples: int) -> JR:
    """Sample random rotations uniformly from SO(3)."""
    # Sample random quaternions from unit ball
    quat = jr.normal(key, shape=(n_samples, 4))
    quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
    return JR.from_quat(quat)


@partial(jax.jit, static_argnames=["n_samples", "step_len"])
def measure_max_angle(key: jr.PRNGKey, n_samples: int, step_len: float) -> float:
    """Measure the maximum angle between random rotations and their euler actions."""
    key1, key2 = jr.split(key)
    rot = sample_random_rotations(key1, n_samples)
    action = jr.uniform(key2, shape=(n_samples, 3), minval=-1, maxval=1)
    rot_after = rel_euler_rotation(rot, action, step_len, scale=True)
    # Compute angle between rotations
    angle = (rot.inv() * rot_after).magnitude()
    return jnp.max(angle) / jnp.pi


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    n_samples = 100_000_000
    step_len = 0.05
    max_angle = measure_max_angle(key, n_samples, step_len)
    print(
        f"Maximum angle for step length {step_len}: {max_angle:.3e} rad/pi, should be {step_len:.1e}."
    )
