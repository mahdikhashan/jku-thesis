# jku-thesis

This project turns Llama-3.2-1B into a "Lizard" model ([arXiv:2507.09025](https://arxiv.org/abs/2507.09025)). Lizard replaces normal softmax attention with two parts: gated linear attention (GLA) and a small sliding-window softmax attention (AWA). We train it in two stages, using the original Llama as the teacher.

## 1. Set up the conda environment

You need [conda](https://docs.conda.io/) (Miniconda is fine). Create the environment once, then turn it on every time you work:

```bash
conda env create -f envs/distill_lizard_llama_3_2_1B.yaml
```

```bash
conda activate distill_lizard_llama_3_2_1B
```

If you change the YAML file later, update the environment with `conda env update -f envs/distill_lizard_llama_3_2_1B.yaml --prune`.

Llama-3.2-1B is a gated model. Ask for access on its Hugging Face page, then log in once:

```bash
hf auth login
```

## 2. Set up Weights & Biases (wandb)

Training sends the loss and some Lizard values to wandb. Make a free account at [wandb.ai](https://wandb.ai), copy your API key from [wandb.ai/authorize](https://wandb.ai/authorize), and log in once:

```bash
wandb login
```

Runs go to the project `lizard-1b` (you can change `WANDB_PROJECT` in `config.py`). If you do not want to use wandb, set `WANDB_MODE=disabled` before the command, for example `WANDB_MODE=disabled python train.py`.

## 3. Run the tests

The tests are small and run on a normal CPU. They check the Lizard math against slow, simple loops, and they check the training code on a tiny model. Install pytest once, then run all tests:

```bash
pip install pytest
```

```bash
pytest
```

- Run one file: `pytest test_lizard.py`
- Run one test: `pytest test_lizard.py::test_gla_matches_formula_loops`
- `test_baselines.py` compares our code with other libraries. It skips a test when a library (like `flash-linear-attention`) is not installed, and it skips GPU tests when there is no GPU.

## 4. Train

You need a CUDA GPU. All settings (learning rates, sequence length, batch size, and so on) are in `config.py`.

```bash
python train.py
```

- Stage 1 trains only the new Lizard parts, so each attention layer copies the output of the teacher's attention layer. The result is saved to `checkpoints/stage1_lizard.pt`.
- Stage 2 fine-tunes the whole model with LoRA on the Alpaca data. The final model is saved to `checkpoints/stage2_model.pt`.

## 5. Evaluate

This runs lm-eval on PIQA, ARC-Easy, ARC-Challenge, HellaSwag, and WinoGrande (0-shot), and on MMLU (5-shot).

```bash
python evaluate.py
```

- With no argument, it tests the trained Lizard model from `checkpoints/stage2_model.pt`.
- To test the original Llama model for comparison, run `python evaluate.py llama`.

## 6. Generate text

This asks a few short questions and prints the answers, so you can compare the original Llama and the trained Lizard model side by side.

```bash
python generate.py
```

- To run only one model, name it: `python generate.py lizard` or `python generate.py llama`.
- The Lizard model needs `checkpoints/stage2_model.pt` from training.

## 7. Benchmark (optional)

`benchmark.py` compares our GLA and AWA code with other versions (FLA, PyTorch SDPA, gpt-oss). It prints an accuracy table, and on a GPU it also prints a speed and memory table.

```bash
python benchmark.py
```

## 8. Kernels (optional)

The `kernels/` folder has separate Triton experiments for fast GLA. They use their own environment:

```bash
conda env create -f envs/gla_kernel_benchmark.yaml
```

```bash
conda activate gla-bench
```

Check that a kernel is correct first, then measure its speed:

```bash
python kernels/gla_triton_flash.py --check
```

```bash
python kernels/gla_triton_benchmark.py
```
