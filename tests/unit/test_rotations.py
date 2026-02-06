import numpy as np
import pytest
from jax.scipy.spatial.transform import Rotation as JR
from scipy.spatial.transform import Rotation as R

from rotations.rotations import RotType, jax_rot_pow, quat_scale


@pytest.mark.unit
def test_quat_scale():
    q1 = R.from_quat([0, 0, 0, 1])
    q2 = R.from_quat([0, 0, 0, -1])
    assert np.allclose((quat_scale(q1, 1.0)).as_quat(), [0, 0, 0, 1])
    assert np.allclose((quat_scale(q1, 0.9)).as_quat(), [0, 0, 0, 1])
    assert np.allclose((quat_scale(q2, 1.0)).as_quat(), [0, 0, 0, -1])
    assert np.allclose((quat_scale(q2, 0.9)).as_quat(), [0, 0, 0, -1])
    q1 = R.from_quat(np.array([1, 0, 0, -1]) / np.sqrt(2))
    scale = 0.5
    # Make sure that this is a case where the quat_scale is not the same as the scipy quat_power
    assert not np.allclose(quat_scale(q1, scale).as_quat(), (q1**scale).as_quat())
    # Check if the sign is preserved
    q_scaled = quat_scale(q1, scale).as_quat()
    assert q_scaled[3] * q1.as_quat()[3] > 0 and q_scaled[0] * q1.as_quat()[0] > 0
    with pytest.raises(AssertionError):
        q_scipy = (q1**scale).as_quat()
        assert q_scipy[3] * q1.as_quat()[3] > 0 and q_scipy[0] * q1.as_quat()[0] > 0


@pytest.mark.unit
def test_quat_scale_batch():
    # Create a batch of quaternions using scipy's batching
    qs = np.array([[0, 0, 0, 1], [0, 0, 0, -1], [1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)]])
    rot = R.from_quat(qs)
    # Test scaling the batch
    scale = 0.5
    q_scaled = quat_scale(rot, scale).as_quat()
    orig_quat = rot.as_quat()
    # Check if signs are preserved for each quaternion in batch
    assert np.all(q_scaled[:, 3] * orig_quat[:, 3] > 0)
    # For vectors that aren't zero, check sign preservation
    nonzero = np.all(orig_quat[:, :3] != 0, axis=1)
    assert np.all((q_scaled[nonzero, :3] * orig_quat[nonzero, :3]) > 0)
    # Check that the function works if all quaternions trigger the masking
    quat_scale(R.from_quat([[1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)]]), 1.0)
    # And that it works if none trigger the masking
    quat_scale(R.from_quat([[0, 0, 0, 1]]), 1.0)


@pytest.mark.unit
def test_jax_rot_pow():
    q1 = JR.from_quat([0, 0, 0, 1])
    q2 = JR.from_quat([0, 0, 0, -1])
    assert np.allclose((jax_rot_pow(q1, 1.0)).as_quat(), [0, 0, 0, 1])
    assert np.allclose((jax_rot_pow(q1, 0.9)).as_quat(), [0, 0, 0, 1])
    assert np.allclose((jax_rot_pow(q2, 1.0)).as_quat(), [0, 0, 0, -1])
    assert np.allclose((jax_rot_pow(q2, 0.9)).as_quat(), [0, 0, 0, -1])
    q1 = JR.from_quat(np.array([1, 0, 0, -1]) / np.sqrt(2))
    scale = 0.5
    # Make sure that this is a case where the quat_scale is not the same as the scipy quat_power
    q_scipy = (R.from_quat(q1.as_quat()) ** scale).as_quat()
    assert not np.allclose(jax_rot_pow(q1, scale).as_quat(), q_scipy)
    # Check if the sign is preserved
    q_scaled = jax_rot_pow(q1, scale).as_quat()
    assert q_scaled[3] * q1.as_quat()[3] > 0 and q_scaled[0] * q1.as_quat()[0] > 0
    with pytest.raises(AssertionError):
        assert q_scipy[3] * q1.as_quat()[3] > 0 and q_scipy[0] * q1.as_quat()[0] > 0


@pytest.mark.unit
def test_jax_rot_pow_batch():
    # Create a batch of quaternions using scipy's batching
    qs = np.array([[0, 0, 0, 1], [0, 0, 0, -1], [1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)]])
    rot = JR.from_quat(qs)
    # Test scaling the batch
    scale = 0.5
    q_scaled = jax_rot_pow(rot, scale).as_quat()
    orig_quat = rot.as_quat()
    # Check if signs are preserved for each quaternion in batch
    assert np.all(q_scaled[:, 3] * orig_quat[:, 3] > 0)
    # For vectors that aren't zero, check sign preservation
    nonzero = np.all(orig_quat[:, :3] != 0, axis=1)
    assert np.all((q_scaled[nonzero, :3] * orig_quat[nonzero, :3]) > 0)
    # Check that the function works if all quaternions trigger the masking
    jax_rot_pow(JR.from_quat([[1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)]]), 1.0)
    # And that it works if none trigger the masking
    jax_rot_pow(JR.from_quat([[0, 0, 0, 1]]), 1.0)


@pytest.mark.unit
def test_r6_recovery():
    n_samples = 100
    rng = np.random.default_rng(0)
    random_rots = R.random(n_samples, random_state=rng)
    mat = random_rots.as_matrix()
    recovered = RotType.r6.from_array(RotType.r6.as_array(random_rots)).as_matrix()
    assert np.allclose(mat, recovered, atol=1e-6), "Inconsistent R6 to matrix recovery"
    # Also check for single rotations
    single_rot = R.random()
    recovered = RotType.r6.from_array(RotType.r6.as_array(single_rot)).as_matrix()
    mat = single_rot.as_matrix()
    assert np.allclose(mat, recovered, atol=1e-6), "Inconsistent single R6 to matrix recovery"
