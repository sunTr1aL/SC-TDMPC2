# TD-MPC2 with Speculative Execution and Learned Corrector

This repository extends the open-source TD-MPC/TD-MPC2 control stack with speculative multi-step execution and a distillation-based corrector that imitates TD-MPC2 replanning. The goal is to accelerate inference by executing several planned actions before replanning while preserving robustness through a lightweight corrector trained from TD-MPC2 rollouts.

The codebase is intended for research on speeding up model-based control with minimal performance loss. It builds on the official TD-MPC/TD-MPC2 releases but is **not** an official implementation of those projects.

-----

## Features
- Single-task TD-MPC2-style training for continuous control tasks (Hydra configuration in `tdmpc2/config.yaml`).
- Speculative multi-step execution of TD-MPC2 plans to reduce replanning frequency.
- Learned corrector trained by distillation from a TD-MPC2 teacher to adjust speculative actions when real states deviate from predictions.
- End-to-end scripts for training the TD-MPC2 teacher, collecting distillation data, training the corrector, and evaluating speculative execution at different horizons.

-----

## Repository layout
- `tdmpc2/tdmpc2/` – TD-MPC2-style agent, speculative execution utilities, corrector implementations, and Hydra configs.
- `tdmpc2/scripts/` – Command-line entry points for corrector data collection and speculative-execution evaluation.
- `tdmpc2/docker/` – Example conda environment (`environment.yaml`) and Dockerfile for running MuJoCo-based tasks.
- `logs/` (created at runtime) – Default location for training/evaluation logs and checkpoints.

-----

## Installation
1. Use Python 3.9+.
2. Create an environment (conda example):
   ```bash
   conda env create -f tdmpc2/docker/environment.yaml
   conda activate tdmpc2
   ```
3. Or with `venv` (install dependencies matching `docker/environment.yaml`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   # Install packages listed in tdmpc2/docker/environment.yaml (dm-control, gymnasium, hydra-core, mujoco, torch, etc.)
   ```
4. Ensure MuJoCo and required control suites are available (e.g., `dm-control`, `mujoco` Python package). Set `MUJOCO_GL=egl` if running headless.

-----

## Using official pretrained TD-MPC2 teachers
You can skip training the teacher from scratch by downloading the official checkpoints (all sizes except the ~1M model):

```bash
cd tdmpc2
python scripts/download_tdmpc2_models.py \
  --output_dir tdmpc2_pretrained
```

- By default the downloader skips the smallest model; pass `--include_smallest` if you explicitly want it.
- Checkpoints are saved as `tdmpc2_pretrained/tdmpc2_<size>.pt` (e.g., `tdmpc2_5m.pt`, `tdmpc2_19m.pt`, `tdmpc2_48m.pt`, `tdmpc2_317m.pt`).
- You can also point `--manifest` to a JSON mapping of `{ "5m": "https://...pt", ... }` to override URLs.

-----

## Training the TD-MPC2 Teacher
Train the single-task TD-MPC2-style agent with Hydra (defaults in `tdmpc2/config.yaml`). Example:
```bash
cd tdmpc2
python tdmpc2/train.py \
  task=humanoid-run \
  model_size=5 \
  steps=1000000 \
  seed=1 \
  device=cuda
```
- `task`/`env` selects the Gym/DMControl-style environment name.
- Checkpoints and logs are stored under `logs/<task>/<seed>/<exp_name>/`.
- Use `device=cpu` to run without CUDA (slower).

-----

## Collecting corrector training data from pretrained teachers
Use the downloaded pretrained TD-MPC2 checkpoints as frozen teachers. The script can target a single model size or iterate over all downloaded sizes (excluding ~1M by default):

**Collect data for walker-run (all model sizes)**
```bash
python scripts/collect_corrector_data.py \
  --task walker-run \
  --all_model_sizes \
  --model_dir tdmpc2_pretrained \
  --episodes 50 \
  --plan_horizon 3 \
  --history_len 4 \
  --output_dir data/walker_run
```

- The default output name is automatically expanded to `data/walker_run/corrector_data_<model_id>.pt`.
- Each dataset stores `z_real`, `z_pred`, `a_plan`, `a_teacher`, `distance`, and `history_feats`.

-----

## Training the corrector
Train both corrector architectures on the collected buffers.
We use a larger architecture (`hidden_dim=512`, `num_layers=4`) and run on a single GPU to avoid NCCL issues.

**Train correctors for walker-run**
```bash
CUDA_VISIBLE_DEVICES=0 python -m tdmpc2.train_corrector \
  --data_dir data/walker_run \
  --corrector_dir correctors/walker_run \
  --corrector_type both \
  --epochs 20 \
  --batch_size 256 \
  --history_len 4 \
  --hidden_dim 512 \
  --num_layers 4 \
  --save_path correctors/walker_run
```

This produces `correctors/walker_run/corrector_<model_id>_two_tower.pth` and `temporal.pth`.

-----

## Evaluating speculative execution
Evaluate the models with the trained correctors using `scripts/eval_mt30_humanoid.py`.

**Evaluate walker-run (all models)**
```bash
python scripts/eval_mt30_humanoid.py \
  --task walker-run \
  --corrector_dir correctors/walker_run \
  --episodes 5 \
  --max_steps 1000 \
  --corrector_hidden_dim 512 \
  --corrector_layers 4
```

**Evaluate a single model size (e.g. 19M)**
```bash
python scripts/eval_mt30_humanoid.py \
  --task walker-run \
  --corrector_dir correctors/walker_run \
  --model_size 19M \
  --episodes 5 \
  --max_steps 1000 \
  --corrector_hidden_dim 512 \
  --corrector_layers 4
```

-----

## Distributed Multi-GPU Training (DDP)
Use the provided DDP tooling to scale TD-MPC2 training across multiple GPUs:

- Quick-start shell script (8 GPUs by default):
  ```bash
  cd tdmpc2
  ./run_ddp_8gpu.sh dog-run 5
  ```
- Direct Python invocation with custom settings:
  ```bash
  cd tdmpc2
  python train_ddp.py task=humanoid-run model_size=19 world_size=4 sync_freq=2 batch_size=256
  ```
- Config-driven launch:
  ```bash
  cd tdmpc2
  python train_ddp.py --config-name=config_ddp
  ```

  See `tdmpc2/DDP_TRAINING_README.md` for detailed guidance on parameters like `world_size`, `sync_freq`, and troubleshooting tips. Rank 0 handles logging and checkpoints; checkpoints remain loadable for single- or multi-GPU use.
  
-----

## Reproducibility and Logging
- Set `seed=<int>` (Hydra for training/evaluation; CLI flag for scripts) to control randomness.
- Training/evaluation logs, videos, and checkpoints default to `logs/<task>/<seed>/<exp_name>/`.
- For stable comparisons, run multiple seeds and report average returns.

-----

## Citing & Acknowledgements
This repository builds on TD-MPC and TD-MPC2 but is an independent extension with speculative execution and a learned corrector.
- TD-MPC code: https://github.com/nicklashansen/tdmpc  |  Paper: “Temporal Difference Learning for Model Predictive Control” (Hansen et al., ICML 2022), arXiv:2203.04955.
- TD-MPC2 code: https://github.com/nicklashansen/tdmpc2  |  Paper: “TD-MPC2: Scalable, Robust World Models for Continuous Control”, arXiv:2310.16828.

-----

## License
This project is licensed under the terms of the existing `LICENSE` file (MIT).
