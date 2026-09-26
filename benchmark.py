import torch

import baselines as bl
from lizard_attention import awa, gla
from reference import awa_inputs, gla_inputs, rel_err

GPU = torch.cuda.is_available()
DEVICE = "cuda" if GPU else "cpu"
LENGTHS = [128, 512, 2048]
WINDOW = 128


def fla_gla(name):
    def run(*x):
        module = "naive" if name.startswith("naive") else "__init__"
        ops = __import__(f"fla.ops.simple_gla.{module}", fromlist=[name])
        return bl.fla_gla(getattr(ops, name), *x)
    return run


GLA = {"ours": gla,
       "fla naive recurrent": fla_gla("naive_recurrent_simple_gla"),
       "fla naive chunk": fla_gla("naive_chunk_simple_gla"),
       "fla chunk kernel (gpu)": fla_gla("chunk_simple_gla")}
AWA = {"ours": lambda *x: awa(*x, WINDOW),
       "pytorch sdpa": lambda *x: bl.pytorch_sdpa_awa(*x, WINDOW),
       "hf gpt-oss": lambda *x: bl.hf_gpt_oss_awa(*x, WINDOW),
       "openai gpt-oss": lambda *x: bl.openai_gpt_oss_awa(*x, WINDOW),
       "fla naive": lambda *x: bl.fla_awa(*x, WINDOW),
       "fla kernel (gpu)": lambda *x: bl.fla_awa(*x, WINDOW, kernel=True)}


def inputs(branch, length, dtype):
    torch.manual_seed(0)
    if branch == "gla":
        x = gla_inputs(length, batch=1, heads=4, features=64, dim=64)
    else:
        x = awa_inputs(length, batch=1, heads=4, dim=64)
    return [t.to(DEVICE, dtype) for t in x]


def accuracy():
    print("| branch | implementation | dtype | "
          + " | ".join(f"L={n}" for n in LENGTHS) + " |")
    print("|---|---|---|" + "---|" * len(LENGTHS))
    for branch, impls in (("gla", GLA), ("awa", AWA)):
        truth = impls["ours"]
        for name, fn in impls.items():
            if "gpu" in name and not GPU:
                continue
            for dtype in (torch.float32, torch.bfloat16):
                try:
                    errors = [rel_err(fn(*inputs(branch, n, dtype)),
                                      truth(*inputs(branch, n, torch.double)))
                              for n in LENGTHS]
                except (ImportError, RuntimeError) as error:
                    print(f"| {branch} | {name} | skipped: {error} |")
                    break
                cells = " | ".join(f"{e:.1e}" for e in errors)
                print(f"| {branch} | {name} | {str(dtype)[6:]} | {cells} |")


def speed(fn, x, backward):
    x = [t.detach().requires_grad_(backward) for t in x]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for step in range(12):
        if step == 2:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start.record()
        out = fn(*x)
        if backward:
            out.sum().backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 10, torch.cuda.max_memory_allocated()


def speed_table(lengths=(512, 2048, 4096), dtype=torch.bfloat16):
    print("| branch | implementation | L | fwd ms | fwd+bwd ms | GiB |")
    print("|---|---|---|---|---|---|")
    for branch, impls in (("gla", GLA), ("awa", AWA)):
        for name in ("ours", "fla chunk kernel (gpu)", "fla kernel (gpu)"):
            if name not in impls:
                continue
            for n in lengths:
                x = inputs(branch, n, dtype)
                fwd, _ = speed(impls[name], x, False)
                both, memory = speed(impls[name], x, True)
                print(f"| {branch} | {name} | {n} | {fwd:.2f} | {both:.2f}"
                      f" | {memory / 2**30:.2f} |")


if __name__ == "__main__":
    accuracy()
    if GPU:
        speed_table()
