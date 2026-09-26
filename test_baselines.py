import pytest
import torch

import baselines as bl
from lizard_attention import awa, gla
from reference import awa_inputs, gla_inputs, rel_err

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
FLA_NAIVE = ["naive_recurrent_simple_gla", "naive_chunk_simple_gla",
             "naive_parallel_simple_gla"]
AWA_CPU = {"pytorch_sdpa": (bl.pytorch_sdpa_awa, None),
           "hf_gpt_oss": (bl.hf_gpt_oss_awa, "transformers"),
           "openai_gpt_oss": (bl.openai_gpt_oss_awa, "gpt_oss"),
           "fla_naive": (bl.fla_awa, "fla")}


@pytest.mark.parametrize("name", FLA_NAIVE)
def test_gla_matches_fla_reference(name):
    naive = pytest.importorskip("fla.ops.simple_gla.naive")
    fq, fk, v, gamma = gla_inputs(300)
    expected = gla(fq, fk, v, gamma)
    got = bl.fla_gla(getattr(naive, name), fq, fk, v, gamma)
    assert rel_err(got, expected) < 1e-5


@pytest.mark.parametrize("name", AWA_CPU)
def test_awa_matches_reference(name):
    fn, package = AWA_CPU[name]
    if package:
        pytest.importorskip(package)
    q, k, v, meta = awa_inputs(300, batch=1)
    assert rel_err(fn(q, k, v, meta, 128), awa(q, k, v, meta, 128)) < 1e-5


@GPU
@pytest.mark.parametrize("dtype, tolerance",
                         [(torch.float32, 1e-3), (torch.bfloat16, 3e-2)])
def test_gla_matches_fla_kernel(dtype, tolerance):
    ops = pytest.importorskip("fla.ops.simple_gla")
    inputs = gla_inputs(2048, batch=1, heads=4, features=64, dim=64)
    x = [t.cuda() for t in inputs]
    got = bl.fla_gla(ops.chunk_simple_gla, *(t.to(dtype) for t in x))
    assert rel_err(got, gla(*x)) < tolerance


@GPU
@pytest.mark.parametrize("dtype, tolerance",
                         [(torch.float32, 1e-3), (torch.bfloat16, 3e-2)])
def test_awa_matches_fla_kernel(dtype, tolerance):
    pytest.importorskip("fla")
    inputs = awa_inputs(2048, batch=1, heads=4, dim=64)
    q, k, v, meta = (t.cuda() for t in inputs)
    got = bl.fla_awa(*(t.to(dtype) for t in (q, k, v, meta)), 128,
                     kernel=True)
    assert rel_err(got, awa(q, k, v, meta, 128)) < tolerance
