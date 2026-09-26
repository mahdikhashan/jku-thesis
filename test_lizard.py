import os

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      LlamaRotaryEmbedding)

import lizard_attention as lz
from reference import (awa_inputs, awa_loop, gla_inputs, gla_loop,
                       gla_recurrent, hedgehog_loop, lizard_loop,
                       random_layer, rel_err, tiny_config)

EXACT = 1e-10


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_hedgehog_matches_formula():
    x, w = torch.randn(2, 3, 20, 16).double(), torch.randn(8, 16).double()
    assert rel_err(lz.hedgehog(x, w), hedgehog_loop(x, w)) < EXACT


def test_hedgehog_halves_are_positive_and_sum_to_one():
    f = lz.hedgehog(torch.randn(2, 3, 20, 16), torch.randn(8, 16))
    assert (f > 0).all()
    assert torch.allclose(f.view(2, 3, 20, 2, 8).sum(-1), torch.ones(1))


@pytest.mark.parametrize("length", [1, 5, 40])
def test_gla_matches_formula_loops(length):
    fq, fk, v, gamma = gla_inputs(length)
    assert rel_err(lz.gla(fq, fk, v, gamma),
                   gla_loop(fq, fk, v, gamma)) < EXACT


@pytest.mark.parametrize("length", [1, 64, 300])
def test_gla_matches_recurrent_form(length):
    fq, fk, v, gamma = gla_inputs(length)
    assert rel_err(lz.gla(fq, fk, v, gamma),
                   gla_recurrent(fq, fk, v, gamma)) < EXACT


def test_gla_gate_zero_keeps_only_the_current_token():
    fq, fk, v, gamma = gla_inputs(30)
    assert rel_err(lz.gla(fq, fk, v, torch.zeros_like(gamma)), v) < EXACT


def test_gla_constant_values_stay_constant():
    fq, fk, v, gamma = gla_inputs(30)
    c = torch.full_like(v, 0.37)
    assert rel_err(lz.gla(fq, fk, c, gamma), c) < EXACT


@pytest.mark.parametrize("length", [1, 15, 16, 17, 100])
def test_awa_matches_formula_loops(length):
    q, k, v, meta = awa_inputs(length)
    assert rel_err(lz.awa(q, k, v, meta, 16),
                   awa_loop(q, k, v, meta, 16)) < EXACT


def test_awa_without_sinks_and_window_is_causal_softmax():
    q, k, v, _ = awa_inputs(40)
    no_sinks = torch.full((4,), -1e4).double()
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert rel_err(lz.awa(q, k, v, no_sinks, 10**6), expected) < EXACT


def test_awa_many_sinks_equal_one_sink_with_logsumexp():
    q, k, v, meta = awa_inputs(40)
    one = torch.logsumexp(meta, 0, keepdim=True)
    assert rel_err(lz.awa(q, k, v, meta, 16),
                   lz.awa(q, k, v, one, 16)) < EXACT


@pytest.mark.parametrize("branch", ["gla", "awa"])
def test_branch_has_zero_gradient_from_future_tokens(branch):
    inputs = gla_inputs(20) if branch == "gla" else awa_inputs(20)
    q, k, v, extra = [t.requires_grad_() for t in inputs]
    out = lz.gla(q, k, v, extra) if branch == "gla" else lz.awa(
        q, k, v, extra, 16)
    out[:, :, 9].sum().backward()
    for grad in (q.grad, k.grad, v.grad):
        assert (grad[:, :, 10:] == 0).all()


def test_awa_has_gradient_only_inside_the_window():
    q, k, v, meta = [t.requires_grad_() for t in awa_inputs(40)]
    lz.awa(q, k, v, meta, 16)[:, :, 39].sum().backward()
    assert (k.grad[:, :, :24] == 0).all()
    assert (k.grad[:, :, 24:].abs().sum(-1) > 0).all()


@pytest.mark.parametrize("fn", [lz.hedgehog, lz.gla, lz.awa])
def test_gradients_match_finite_differences(fn):
    if fn is lz.hedgehog:
        args = (torch.randn(1, 2, 4, 5), torch.randn(3, 5))
    elif fn is lz.gla:
        args = gla_inputs(6, batch=1, heads=2, features=3, dim=2)
    else:
        args = awa_inputs(6, batch=1, heads=2, dim=3)
    args = [a.double().requires_grad_() for a in args]
    call = fn if fn is not lz.awa else (lambda *a: lz.awa(*a, 3))
    assert torch.autograd.gradcheck(call, args)


def test_layer_matches_formula_loops():
    layer = random_layer()
    x = torch.randn(2, 20, 128).double()
    assert rel_err(layer(x)[0], lizard_loop(layer, x)) < EXACT


def test_layer_is_causal_and_batch_independent():
    layer = random_layer()
    x = torch.randn(3, 30, 128).double().requires_grad_()
    y = layer(x)[0]
    single = torch.cat([layer(x[i:i + 1])[0] for i in range(3)])
    assert rel_err(y, single) < EXACT
    y[:, 9].sum().backward()
    assert (x.grad[:, 10:] == 0).all()


def test_every_lizard_parameter_gets_a_gradient():
    layer = lz.LizardAttention(tiny_config(), 0, window=16, feature_dim=8)
    y, _ = layer(torch.randn(2, 30, 128))
    (y * torch.randn_like(y)).sum().backward()
    for name in ("phi_q", "phi_k", "W_gamma", "meta_tokens", "alpha_blend"):
        grad = [p.grad for n, p in layer.named_parameters() if name in n][0]
        assert torch.isfinite(grad).all() and grad.abs().max() > 0, name


def softmax_only(layer, monkeypatch):
    monkeypatch.setattr(lz, "gla", lambda fq, fk, v, g: torch.zeros_like(v))
    layer.window = 10**6
    with torch.no_grad():
        layer.meta_tokens.fill_(-1e4)
    return layer


def llama_output(llama, x, rope):
    mask = torch.full((x.shape[1], x.shape[1]), float("-inf")).triu(1)
    out, _ = llama(hidden_states=x, position_embeddings=rope,
                   attention_mask=mask.to(x.dtype)[None, None])
    return out


def test_equals_llama_attention_in_the_softmax_limit(monkeypatch):
    config = tiny_config()
    llama = LlamaAttention(config, 0).double()
    lizard = softmax_only(lz.from_llama(llama, config), monkeypatch)
    x = torch.randn(2, 30, 128).double()
    no_rope = (torch.ones(2, 30, 32).double(),
               torch.zeros(2, 30, 32).double())
    rope = LlamaRotaryEmbedding(config)(x, torch.arange(30)[None])
    assert rel_err(lizard(x)[0], llama_output(llama, x, no_rope)) < 1e-6
    assert rel_err(lizard(x)[0], llama_output(llama, x, rope)) > 1e-3


def test_whole_model_logits_unchanged_after_swap(monkeypatch):
    name = os.environ.get("LIZARD_TEST_MODEL")
    if name:
        model = AutoModelForCausalLM.from_pretrained(name).eval()
    else:
        model = LlamaForCausalLM(tiny_config(layers=2)).eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    monkeypatch.setattr(
        model.model.rotary_emb, "forward",
        lambda x, position_ids: (torch.ones(*position_ids.shape, head_dim),
                                 torch.zeros(*position_ids.shape, head_dim)))
    ids = torch.randint(0, model.config.vocab_size, (1, 64))
    expected = model(ids).logits
    for block in model.model.layers:
        block.self_attn = softmax_only(
            lz.from_llama(block.self_attn, model.config), monkeypatch)
    assert rel_err(model(ids).logits, expected) < 1e-4
