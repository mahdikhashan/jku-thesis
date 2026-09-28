# jku-thesis

This project turns Llama-3.2-1B into a [Lizard](https://arxiv.org/abs/2507.09025) model. Lizard replaces softmax attention with gated linear attention plus a small sliding-window attention.

## Setup

Create and turn on the conda environment:

```bash
conda env create -f envs/distill_lizard_llama_3_2_1B.yaml
```

```bash
conda activate distill_lizard_llama_3_2_1B
```

Llama-3.2-1B needs access on Hugging Face. Ask for access on the model page, then log in:

```bash
hf auth login
```

Training logs to wandb. Get your key from [wandb.ai/authorize](https://wandb.ai/authorize) and log in:

```bash
wandb login
```

To run without wandb, put `WANDB_MODE=disabled` before the command.

## Test

The tests run on a normal CPU.

```bash
pip install pytest
```

```bash
pytest
```

To run one test: `pytest test_lizard.py::test_gla_matches_formula_loops`

## Train

Needs a GPU. Settings are in `config.py`. The model is saved in `checkpoints/`.

```bash
python train.py
```

## Evaluate

Tests the trained model on common benchmarks. Add `llama` to test the original model instead.

```bash
python evaluate.py
```

## Generate

Prints answers from the original model and the trained model. Add `llama` or `lizard` to run only one.

```bash
python generate.py
```

## Benchmark

Compares our attention code with other libraries for accuracy and speed.

```bash
python benchmark.py
```

## Kernels

The `kernels/` folder has Triton experiments with their own environment:

```bash
conda env create -f envs/gla_kernel_benchmark.yaml
```

```bash
conda activate gla-bench
```

```bash
python kernels/gla_triton_flash.py --check
```
