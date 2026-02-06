import pytest
import torch
from scipy.spatial.transform import Rotation as R

from rotations.projections import project_to_quat_plus


@pytest.mark.unit
def test_quatplus_proj():
    # Test that all negative quats are projected to their positive counterparts
    # and that the positive quats are left unchanged
    q1_neg = torch.tensor([0, 0, 0, -1.0])
    q1_pos = -q1_neg
    q2_neg = torch.tensor([0, 0, -1.0, 0])
    q2_pos = -q2_neg
    q3_neg = torch.tensor([0, -1.0, 0, 0])
    q3_pos = -q3_neg
    q4_neg = torch.tensor([-1.0, 0, 0, 0])
    q4_pos = -q4_neg
    assert torch.allclose(project_to_quat_plus(q1_neg), q1_pos)
    assert torch.allclose(project_to_quat_plus(q1_pos), q1_pos)
    assert torch.allclose(project_to_quat_plus(q2_neg), q2_pos)
    assert torch.allclose(project_to_quat_plus(q2_pos), q2_pos)
    assert torch.allclose(project_to_quat_plus(q3_neg), q3_pos)
    assert torch.allclose(project_to_quat_plus(q3_pos), q3_pos)
    assert torch.allclose(project_to_quat_plus(q4_neg), q4_pos)
    assert torch.allclose(project_to_quat_plus(q4_pos), q4_pos)

    # Test that all projected quats are valid unit quaternions
    q1 = torch.as_tensor(R.random().as_quat(), dtype=torch.float32)
    q2 = torch.as_tensor(R.random().as_quat(), dtype=torch.float32)
    q3 = torch.as_tensor(R.random().as_quat(), dtype=torch.float32)
    q4 = torch.as_tensor(R.random().as_quat(), dtype=torch.float32)
    assert torch.allclose(project_to_quat_plus(q1).norm(), torch.ones(1))
    assert torch.allclose(project_to_quat_plus(q2).norm(), torch.ones(1))
    assert torch.allclose(project_to_quat_plus(q3).norm(), torch.ones(1))
    assert torch.allclose(project_to_quat_plus(q4).norm(), torch.ones(1))


@pytest.mark.unit
def test_quat_scale_batch():
    # Test negative and positive quats when batched
    q_neg = torch.tensor([[0, 0, 0, -1.0], [0, 0, -1.0, 0], [0, -1.0, 0, 0], [-1.0, 0, 0, 0]])
    q_pos = -q_neg
    assert torch.allclose(project_to_quat_plus(q_neg), q_pos)
    assert torch.allclose(project_to_quat_plus(q_pos), q_pos)

    # Test norm when quats are batched
    q = torch.as_tensor(R.random(100).as_quat(), dtype=torch.float32)
    assert torch.allclose(project_to_quat_plus(q).norm(dim=-1), torch.ones(100))
