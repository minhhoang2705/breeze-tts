import torch

from breeze_omni.components.sampling import apply_repetition_penalty, sample


def test_mixed_greedy_and_top_k_sampling():
    logits = torch.tensor([[1.0, 3.0, 2.0], [1.0, 3.0, 2.0]])
    tokens = sample(logits, torch.tensor([0.0, 1.0]), torch.tensor([1, 1]), torch.tensor([1.0, 1.0]), generator=torch.Generator().manual_seed(7))
    assert tokens.tolist() == [1, 1]


def test_repetition_penalty_is_rowwise_and_hf_signed():
    logits = torch.tensor([[2.0, -2.0, 0.5], [2.0, -2.0, 0.5]])
    mask = torch.tensor([[True, True, False], [False, True, True]])
    result = apply_repetition_penalty(logits.clone(), mask, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(result, torch.tensor([[1.0, -4.0, 0.5], [2.0, -8.0, 0.125]]))


def test_scalar_parameters_broadcast():
    logits = torch.tensor([[0.0, 2.0], [3.0, 1.0]])
    result = sample(logits, torch.tensor(0.0), torch.tensor(50), torch.tensor(1.0))
    assert result.tolist() == [1, 0]
