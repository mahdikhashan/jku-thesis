import pytest
import torch
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM

import config as C
import lizard_attention
import train
from reference import tiny_config


@pytest.fixture(autouse=True)
def small(monkeypatch):
    monkeypatch.setattr(C, "DEVICE", "cpu")
    monkeypatch.setattr(C, "GRAD_ACCUM", 2)
    monkeypatch.setattr(C, "EPOCHS", 1)
    monkeypatch.setattr(C, "LIZARD", {"window": 8, "feature_dim": 8})


def tiny_llama():
    torch.manual_seed(0)
    return LlamaForCausalLM(tiny_config(layers=2)).eval()


def tiny_loader():
    torch.manual_seed(0)
    return DataLoader(torch.randint(0, 100, (6, 32)), batch_size=1)


def test_is_lizard_matches_only_lizard_parameters():
    assert train.is_lizard("model.layers.0.self_attn.phi_q.weight")
    assert train.is_lizard("base_model.model.layers.1.self_attn.alpha_blend")
    assert not train.is_lizard("model.layers.0.mlp.gate_proj.weight")
    assert not train.is_lizard("model.layers.0.self_attn.q_proj.weight")


def test_pack_and_learning_rate_schedule():
    packed = train.pack([[1, 2, 3], [4, 5], [6, 7, 8, 9]], 4)
    assert torch.equal(packed, torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
    assert train.lr_factor(0, 100) == 0
    assert train.lr_factor(10, 100) == pytest.approx(1.0)
    assert train.lr_factor(100, 100) == pytest.approx(C.MIN_LR_RATIO)


def test_stage1_trains_alpha_and_sinks():
    state = train.stage1(tiny_llama(), tiny_loader())
    _, unexpected = train.to_lizard(tiny_llama()).load_state_dict(
        state, strict=False)
    assert not unexpected
    assert state["model.layers.0.self_attn.alpha_blend"].item() != 1.0
    assert state["model.layers.0.self_attn.meta_tokens"].abs().max() > 0


def test_stage2_merges_lora_and_reloads_strictly():
    state = train.stage1(tiny_llama(), tiny_loader())
    saved = train.stage2(tiny_llama(), state, tiny_loader()).state_dict()
    assert not any("lora" in key for key in saved)
    train.to_lizard(tiny_llama()).load_state_dict(saved, strict=True)
    before = tiny_llama().model.layers[0].self_attn.q_proj.weight
    assert not torch.equal(saved["model.layers.0.self_attn.q_proj.weight"],
                           before)


def test_gradient_checkpointing_recomputes_without_changing_results(
        monkeypatch):
    state = train.stage1(tiny_llama(), tiny_loader())
    forward = lizard_attention.LizardAttention.forward
    results = []
    for checkpointing in (False, True):
        calls = []
        monkeypatch.setattr(C, "GRADIENT_CHECKPOINTING", checkpointing)
        monkeypatch.setattr(
            lizard_attention.LizardAttention, "forward",
            lambda self, *a, **k: calls.append(1) or forward(self, *a, **k))
        model = train.stage2(tiny_llama(), state, tiny_loader())
        results.append((len(calls), model.state_dict()))
    (calls_a, weights_a), (calls_b, weights_b) = results
    assert calls_b == 2 * calls_a
    assert all(torch.allclose(weights_a[k], weights_b[k]) for k in weights_a)
