import numpy as np
import pytest
import torch
from scipy.fft import dctn
from torch import nn
from freqblender.frequency import DCT2, FPNet, frequency_priors
from freqblender.losses import parser_loss, counterfactual_views, secant_loss, distillation_kl
from freqblender.models import freeze


@pytest.mark.parametrize("size", [32, 64, 224, 400])
def test_dct_reference_reconstruction_and_energy(size):
    torch.manual_seed(7)
    x = torch.rand(1, 3, size, size)
    transform = DCT2(size)
    c = transform(x)
    expected = dctn(x.numpy(), type=2, axes=(-2, -1), norm="ortho")
    np.testing.assert_allclose(c.numpy(), expected, atol=8e-4, rtol=3e-4)
    torch.testing.assert_close(transform.inverse(c), x, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(c.square().sum(), x.square().sum(), atol=.1, rtol=2e-5)


def test_prior_partition_and_boundaries():
    p = frequency_priors(33)
    assert torch.equal(p.sum(1), torch.ones(1, 33, 33))
    assert p[0, 0, 0, 2] == 1   # exactly 1/16
    assert p[0, 1, 0, 3] == 1
    assert p[0, 1, 0, 16] == 1  # exactly 1/2
    assert p[0, 2, 0, 17] == 1


def test_blend_uses_fake_masks_without_complement_shortcut():
    parser = FPNet(32, 2)
    r, f = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        c, masks = parser(f)
        expected = parser.dct.inverse(parser.dct(r)*(masks[:, 0:1]+masks[:, 2:3]) + c*masks[:, 1:2])
        actual = parser.blend(r, f, clamp=False)
    torch.testing.assert_close(actual, expected)
    assert (masks.sum(1)-1).abs().mean() > .05  # independent sigmoids, not softmax


class TinyIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 2))

    def forward(self, x):
        return self.net(x)


def test_parser_all_four_losses_and_frozen_auxiliary_gradients():
    parser = FPNet(32, 2)
    identity, detector = freeze(TinyIdentity()), freeze(TinyDetector())
    real, fake = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    loss, terms = parser_loss(parser, real, fake, identity, detector)
    assert set(terms) == {"ff", "ad", "qa", "pi"}
    assert all(torch.isfinite(value) for value in terms.values())
    # Each auxiliary objective individually reaches the parser's appropriate head.
    grad_ff = torch.autograd.grad(terms["ff"], parser.decoders[0][0].weight, retain_graph=True)[0]
    grad_ad = torch.autograd.grad(terms["ad"], parser.decoders[1][0].weight, retain_graph=True)[0]
    assert grad_ff.abs().sum() > 0 and grad_ad.abs().sum() > 0
    loss.backward()
    for head in parser.decoders:
        assert head[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in identity.parameters())
    assert all(p.grad is None for p in detector.parameters())


def test_bounded_counterfactuals_and_zero_donor():
    parser = freeze(FPNet(32, 2))
    x, donor = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    for view in counterfactual_views(x, donor, parser):
        assert view.min() >= 0 and view.max() <= 1
        assert ((view-x).square().mean((1, 2, 3)).sqrt() <= .005001).all()
        assert not view.requires_grad
    for unchanged in counterfactual_views(x, x, parser):
        torch.testing.assert_close(unchanged, x)


def test_secant_matching_zero_loss_and_confidence_gate():
    x = torch.rand(2, 3, 32, 32)
    xs, xn = x+.005, x+.002
    labels = torch.tensor([0, 1])
    t0 = torch.tensor([[3., -3.], [-3., 3.]])
    ts = t0 + torch.tensor([[.1, .2], [.2, .1]])
    tn = t0.clone()
    loss = secant_loss(t0, ts, tn, t0, ts, tn, labels, x, xs, xn)
    assert loss == 0
    s = ts.clone().requires_grad_(True)
    imperfect = secant_loss(t0, s+1*torch.tensor([[0., 1.]]), tn, t0, ts, tn, labels, x, xs, xn)
    imperfect.backward()
    assert s.grad.abs().sum() > 0
    unreliable = secant_loss(t0, ts+1, tn+5, -t0, -ts, -tn, labels, x, xs, xn)
    assert unreliable == 0
    assert abs(float(distillation_kl(t0, t0))) < 1e-6


def test_degenerate_interventions_are_finite():
    x = torch.zeros(2, 3, 32, 32)
    s = torch.randn(2, 2, requires_grad=True)
    t = torch.randn(2, 2)
    loss = secant_loss(s, s, s, t, t, t, torch.tensor([0, 1]), x, x, x)
    assert torch.isfinite(loss) and loss == 0
