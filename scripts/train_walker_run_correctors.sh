#!/bin/bash
# Train correctors for walker-run (all models found in data_dir)
# Uses single GPU to avoid NCCL errors with DataParallel
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
