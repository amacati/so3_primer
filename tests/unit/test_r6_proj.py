import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from rotations.rotations import r6_2mat, r6_orthonomalization


def _test_matrix(mats: NDArray, tol: float = 1e-5) -> bool:
    """Validate whether a batch of matrices are valid rotation matrices.

    Args:
        mats : (B, 9) Array of batched flattened rotation matrices.
        tol: Numerical tolerance for orthogonality and determinant checks.

    Returns:
        True if all matrices are valid rotation matrices, False otherwise.
    """
    assert mats.ndim == 2 and mats.shape[-1] == 9
    dtype = mats.dtype
    mats = mats.reshape(-1, 3, 3)
    
    # Verify that the transpose is the inverse of the matrices
    prod = mats @ mats.transpose(0, 2, 1)
    EYE = np.tile(np.eye(3, dtype=dtype), reps=(mats.shape[0], 1, 1))
    valid_transpose = np.allclose(prod, EYE, atol=tol)

    # Verify that determinant is approximately 1
    det = np.linalg.det(mats)
    valid_det = np.allclose(det, np.ones_like(det), atol=tol)
    
    valid = valid_transpose and valid_det
    return valid


def _test_r6(vecs: NDArray, tol: float = 1e-5) -> bool:
    """Validate whether a batch of 6-dimensional vectors are made up of two orthonormal axes.

    Args:
        vecs : (B, 6) Array of batched flattened r6 vectors.
        tol: Numerical tolerance for orthogonality and determinant checks.

    Returns:
        True if all vectors contain valid orthonormal vectors, False otherwise.
    """
    assert vecs.ndim == 2 and vecs.shape[-1] == 6
    
    # Concatenate 3rd normal vector and pass them through the matrix checker
    v3 = np.cross(vecs[..., :3], vecs[..., 3:])
    mats = np.concatenate([vecs, v3], axis=-1)
    valid = _test_matrix(mats, tol=tol)
    return valid


@pytest.mark.unit
def test_matrix_validation():
    """Test the validation function for matrices with true and random matrices."""
    # True matrices should pass the test
    rots = R.random(1000)
    mats = rots.as_matrix().reshape(-1, 9)
    assert _test_matrix(mats)

    # Random flattened matrices should not pass the test
    random = np.random.uniform(-1, 1, size=(1000, 9)).astype(np.float32)
    assert not _test_matrix(random)


@pytest.mark.unit
def test_r6_validation():
    """Test the validation function for r6 vectors with true orthonormal and random vectors."""
    # True r6 vectors from matrices should pass the test
    rots = R.random(1000)
    vecs = rots.as_matrix().reshape(-1, 9)[:, :6]
    assert _test_r6(vecs)

    # Random 6-dimensional vectors should not pass the test
    random = np.random.uniform(-1, 1, size=(1000, 6)).astype(np.float32)
    assert not _test_r6(random)


@pytest.mark.unit
def test_r6_2mat():
    """Test the PyTorch and JAX implementations of the r6_2mat() projection.
    
    Note: NumPy implementation dispatches to PyTorch so it's omitted.
    """
    # Test PyTorch version (returns flattened matrices)
    vecs = torch.rand(size=(1000, 6)) * 2 - 1
    mats = r6_2mat(vecs)
    assert _test_matrix(mats.numpy())

    # Test JAX version (returns non-flattened matrices)
    vecs = jax.random.uniform(
        jax.random.key(0), shape=(1000, 6), dtype=jnp.float32, minval=-1, maxval=1
    )
    mats = r6_2mat(vecs).reshape(-1, 9)
    assert _test_matrix(np.asarray(mats))


@pytest.mark.unit
def test_r6_orthonomalization():
    """Test the PyTorch implementation of the r6_orthonomalization() projection.
    
    Note: NumPy implementation dispatches to PyTorch so it's omitted.
    """
    # Test PyTorch version (returns flattened vectors)
    vecs = torch.rand(size=(1000, 6)) * 2 - 1
    vecs = r6_orthonomalization(vecs)
    assert _test_r6(vecs.numpy())
