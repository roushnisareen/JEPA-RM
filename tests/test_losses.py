import math

import pytest
import torch
import torch.nn.functional as F

from jepa_rm.data import PreferencePair
from jepa_rm.rewards.bradley_terry import LengthPenalizedRM, bt_loss, length_slope
from jepa_rm.rewards.embeddings import Standardizer, embed_pairs, pool
from jepa_rm.rewards.frozen_bt import build_head
from jepa_rm.rewards.jepa import build_predictor, jepa_loss


def test_bt_loss_is_log2_when_rewards_tie():
    assert math.isclose(bt_loss(torch.zeros(4), torch.zeros(4)).item(), math.log(2), rel_tol=1e-6)


def test_bt_loss_falls_as_the_margin_grows():
    small = bt_loss(torch.tensor([1.0]), torch.tensor([0.0]))
    large = bt_loss(torch.tensor([5.0]), torch.tensor([0.0]))
    assert large < small


def test_jepa_loss_prefers_predictions_near_the_chosen_answer():
    chosen = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    rejected = F.normalize(torch.tensor([[0.0, 1.0]]), dim=-1)
    good = jepa_loss(chosen, chosen, rejected, tau=0.05)
    bad = jepa_loss(rejected, chosen, rejected, tau=0.05)
    assert good < bad


def test_jepa_loss_gradient_moves_prediction_toward_chosen():
    chosen = F.normalize(torch.randn(8, 16), dim=-1)
    rejected = F.normalize(torch.randn(8, 16), dim=-1)
    raw = torch.randn(8, 16, requires_grad=True)
    jepa_loss(F.normalize(raw, dim=-1), chosen, rejected, tau=0.05).backward()
    with torch.no_grad():
        stepped = F.normalize(raw - 0.1 * raw.grad, dim=-1)
        before = (F.normalize(raw, dim=-1) * chosen).sum(-1).mean()
        after = (stepped * chosen).sum(-1).mean()
    assert after > before


def test_predictor_keeps_the_embedding_size():
    assert build_predictor(dim=32, hidden=64)(torch.randn(5, 32)).shape == (5, 32)


def test_standardizer_gives_zero_mean_unit_std():
    x = torch.randn(1000, 4) * 3 + 7
    z = Standardizer.fit(x)(x)
    assert torch.allclose(z.mean(0), torch.zeros(4), atol=1e-4)
    assert torch.allclose(z.std(0), torch.ones(4), atol=1e-3)


def padded_batch():
    # two sequences, the second padded after 2 real tokens
    hidden = torch.tensor([[[1.0], [2.0], [3.0]], [[10.0], [20.0], [99.0]]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    return hidden, mask


class FakeEncoder:
    """A fixed random vector per text, shifted by 1000 if the text contains any of `shifted`."""

    device = "cpu"

    def __init__(self, shifted: set[str] = frozenset()):
        self.shifted = shifted

    def embed(self, texts):
        vecs = [torch.randn(4, generator=torch.Generator().manual_seed(sum(map(ord, t)))) for t in texts]
        return torch.stack([v + 1000.0 * any(s in t for s in self.shifted) for v, t in zip(vecs, texts, strict=True)])


def test_normalization_is_fitted_on_the_fit_split_only():
    pairs = [PreferencePair(f"<q{i:02d}>", f"<good{i:02d}>", f"<bad{i:02d}>") for i in range(20)]
    clean = embed_pairs(FakeEncoder(), pairs, holdout=0.25)
    held = [pairs[i] for i in clean.hold.tolist()]
    shifted = {t for p in held for t in (p.prompt, p.chosen, p.rejected)}
    # the held-out prompts and answers now embed far away from everything else...
    moved = embed_pairs(FakeEncoder(shifted), pairs, holdout=0.25)
    # ...but they don't take part in the normalization statistics, so those are unchanged
    for a, b in ((clean.ctx_norm, moved.ctx_norm), (clean.tgt_norm, moved.tgt_norm)):
        assert torch.allclose(a.mean, b.mean) and torch.allclose(a.std, b.std)


def test_mean_pooling_ignores_padding():
    hidden, mask = padded_batch()
    assert pool(hidden, mask, "mean").squeeze(-1).tolist() == [2.0, 15.0]


def test_last_pooling_takes_the_last_real_token():
    hidden, mask = padded_batch()
    assert pool(hidden, mask, "last").squeeze(-1).tolist() == [3.0, 20.0]


def test_mean_pooling_is_blind_to_repetition():
    # the design property behind the pooling ablation (in a real causal LM, the last token of the
    # repeated text has attended to both copies, so "last" pooling can tell them apart)
    once = torch.tensor([[[1.0], [3.0]]])
    twice = torch.tensor([[[1.0], [3.0], [1.0], [3.0]]])
    assert torch.equal(pool(once, torch.ones(1, 2), "mean"), pool(twice, torch.ones(1, 4), "mean"))


def test_unknown_pooling_is_rejected():
    hidden, mask = padded_batch()
    with pytest.raises(ValueError):
        pool(hidden, mask, "max")


def test_loss_weights_switch_the_terms_off():
    chosen = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    rejected = F.normalize(torch.tensor([[0.0, 1.0]]), dim=-1)
    pred = chosen
    jepa_only = jepa_loss(pred, chosen, rejected, tau=0.05, rank_weight=0.0)
    rank_only = jepa_loss(pred, chosen, rejected, tau=0.05, jepa_weight=0.0)
    assert jepa_only.item() == pytest.approx(0.0, abs=1e-6)  # prediction already equals chosen
    assert rank_only.item() == pytest.approx(F.softplus(torch.tensor(-1 / 0.05)).item())
    assert jepa_loss(pred, chosen, rejected, tau=0.05).item() == pytest.approx(jepa_only.item() + rank_only.item())


def test_frozen_control_has_the_same_capacity_as_the_jepa_predictor():
    def n_params(module):
        return sum(p.numel() for p in module.parameters())

    dim, hidden = 64, 128
    predictor, head = build_predictor(dim, hidden), build_head(dim, hidden)
    # same weights; the head only adds its single output unit's bias, the predictor dim - 1 more biases
    assert abs(n_params(predictor) - n_params(head)) <= dim + hidden
    assert head(torch.randn(3, 2 * dim)).shape == (3, 1)


def test_length_slope_is_the_least_squares_fit():
    lengths = [10, 20, 30, 40]
    assert length_slope([1 + 0.5 * x for x in lengths], lengths) == pytest.approx(0.5)
    assert length_slope([3.0, 3.0], [5, 5]) == 0.0  # no length variation, nothing to remove


class FakeBT:
    def score(self, prompts, responses):
        return [2.0 + len(r.split()) for r in responses]  # pays exactly 1 per word


def test_length_penalty_removes_the_length_trend():
    rm = LengthPenalizedRM(FakeBT(), slope=1.0)
    assert rm.score(["q", "q"], ["one", "one two three four"]) == [2.0, 2.0]
