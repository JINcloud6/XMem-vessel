import torch

from model.memory_util import get_similarity, do_softmax


def test_spatial_decay_prefers_matching_position():
    mk = torch.zeros(1, 1, 3)
    qk = torch.zeros(1, 1, 1)
    mem_pos = torch.tensor([0.0, 5.0, 10.0])

    similarity = get_similarity(
        mk,
        ms=None,
        qk=qk,
        qe=None,
        mem_pos=mem_pos,
        query_pos=torch.tensor(5.0),
        sigma=8.0,
        lambda_pos=1.0,
    )
    affinity = do_softmax(similarity)

    assert affinity[0, 1, 0] > affinity[0, 0, 0]
    assert affinity[0, 1, 0] > affinity[0, 2, 0]


def test_spatial_decay_sigma_controls_peakedness():
    mk = torch.zeros(1, 1, 3)
    qk = torch.zeros(1, 1, 1)
    mem_pos = torch.tensor([0.0, 5.0, 10.0])

    similarity = get_similarity(
        mk,
        ms=None,
        qk=qk,
        qe=None,
        mem_pos=mem_pos,
        query_pos=5.0,
        sigma=0.5,
        lambda_pos=1.0,
    )
    affinity = do_softmax(similarity)

    assert affinity[0, 1, 0] > 0.9
