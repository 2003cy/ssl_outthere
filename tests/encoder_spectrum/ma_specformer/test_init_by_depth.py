import torch
import torch.nn as nn

from encoder_spectrum.ma_specformer.model.modules import LayerNorm, _init_by_depth


def test_init_by_depth_does_not_touch_layernorm_gamma():
    ln = LayerNorm(dim=16, bias=True)
    gamma_before = ln.weight.detach().clone()
    beta_before = ln.bias.detach().clone()

    _init_by_depth(ln, depth_frac=1.0 / 6)

    # LayerNorm parameters must remain unchanged.
    assert torch.equal(ln.weight, gamma_before)
    assert torch.equal(ln.bias, beta_before)
    assert torch.allclose(ln.weight, torch.ones_like(ln.weight))


def test_init_by_depth_initializes_linear_only():
    linear = nn.Linear(16, 16, bias=True)
    weight_before = linear.weight.detach().clone()
    bias_before = linear.bias.detach().clone()

    _init_by_depth(linear, depth_frac=1.0 / 6)

    assert not torch.equal(linear.weight, weight_before)
    assert not torch.equal(linear.bias, bias_before)
    assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))
