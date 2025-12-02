---
description: This document is for a code assistant working on the [`SC-TDMPC2`](https://github.com/sunTr1aL/SC-TDMPC2/tdmpc2) repo.  It describes **what is already implemented** and how the system is supposed to behave, so new code stays compatible.
---

# 1. High-Level Overview

Our project augments the original **TD-MPC2** (a model-based RL controller with latent-space MPC) by introducing:

### **1. Speculative multi-step execution**
The teacher normally replans at every step.  
We instead allow *open-loop execution* of the entire TD-MPC2 3-step plan — or even extending beyond 3 steps (5 or 6 steps) using the world model.

This speeds up inference significantly but introduces compounding prediction errors.

### **2. Corrector models**
To address mismatch between predicted latent trajectories and real latent encodings when rolling forward open-loop, we introduce two corrector components:

- **Two-Tower Gated MLP Corrector**
- **Temporal Transformer Corrector**

A corrector predicts what TD-MPC2 *would* output if it replanned from the current real state, but does so using fast neural inference instead of MPC.

Given:

- real latent `z_real`
- predicted latent `z_pred`
- planned action `a_plan`
- optional mismatch history `history_feats`

the corrector outputs:

```
a_corr = a_plan + Δa
```

### **3. Distillation training**
The corrector learns by **imitating** the teacher’s true replanning actions (“distillation”).

### **4. Zero training of TD-MPC2**
We do **not** train TD-MPC2 from scratch — it is slow.  
Instead we use downloaded pretrained checkpoints (except the 1M model).

### **5. Multi-step evaluation**
We evaluate:

- baseline (replan every step)
- naive 2-step, 3-step, 5-step, 6-step execution
- corrected 2/3/5/6-step execution using  
  **Two-Tower** and **Temporal** correctors

on the **humanoid-run** task.

---

# 2. Repository Structure

```
sc-tdmpc2/
  tdmpc2/
    common/
    envs/
    corrector.py
    utils_ckpt.py
    ...
  tdmpc2_pretrained/
    mt30-5M.pt
    mt30-19M.pt
    mt30-48M.pt
    mt30-317M.pt
    ...
  correctors/
    corrector_mt30-5M_two_tower.pth
    corrector_mt30-5M_temporal.pth
    ...
  data/
    corrector_data_mt30-5M.pt
    ...
  scripts/
    collect_corrector_data.py
    train_corrector.py
    eval_mt30_humanoid.py   # new small evaluation script
```

### **Model IDs**
We define:

```
model_id = <checkpoint_filename_without_extension>
```

Example:

```
mt30-5M.pt → model_id="mt30-5M"
```

Correctors and datasets follow this naming.

---

# 3. TD-MPC2 Teacher + Speculative Execution

TD-MPC2 produces for each observation:

- latent `z_t`
- planned actions `[a0, a1, a2]`
- predicted latents `[z0, z1_pred, z2_pred, z3_pred]`

The original algorithm executes **only `a0`**, then replans.

We instead support blocks of:

```
exec_horizon = 2, 3, 5, 6
```

### **Relative step logic**

For relative step `k` inside the block:

#### k = 0
Execute `a_plan[0]` directly.

#### k = 1, 2
- naive: `a_plan[k]`
- corrected: `corrector(z_real_k, z_pred_k, a_plan_k, history)`

#### k > 2
We extend:

- predicted latent using world model rollouts  
- planned action by clamping at `a_plan[2]`  
- optionally use corrector to adjust

This enables speculative execution beyond the teacher’s planning horizon.

---

# 4. Corrector Models

Correctors operate on latent mismatches and predicted actions.

Both follow the interface:

```python
a_corr = corrector(
    z_real,
    z_pred,
    a_plan,
    mismatch_history=None
)
```

---

## 4.1 Two-Tower Corrector (Gated MLP)

Inputs:

```
z_real, z_pred, delta_z = z_real - z_pred, a_plan
```

Architecture:

- 4 encoders:
  - MLP_real(z_real)
  - MLP_pred(z_pred)
  - MLP_delta(delta_z)
  - MLP_a(a_plan)
- Concatenate → fused vector
- Gated fusion layer
- Output residual action update `Δa`

Produces:

```
a_corr = a_plan + Δa
```

---

## 4.2 Temporal Transformer Corrector

Uses a history window of **K mismatch feature vectors**:

```
feat_t = [z_real, z_pred, z_real - z_pred, a_plan]
```

Mechanism:

- Project features → tokens
- Add positional encodings
- Transformer encoder (1–2 layers)
- Pool to produce context vector
- Fuse with `a_plan` using gating
- Output `a_corr`

Temporal model captures **drift patterns** across multiple mismatch steps.

---

# 5. Distillation Data Collection

Script: `scripts/collect_corrector_data.py`

For each teacher step:

1. Encode observation → latent `z_real_t`
2. Compute TD-MPC2 3-step plan + predicted latents
3. Roll the environment one step using `a_plan[0]`
4. Encode next observation → `z_real_{t+1}`
5. Save data:

```
z_real_{t+1}
z_pred (predicted latent for step 1)
a_plan[1]
a_teacher = TD-MPC2 replanning action at real state
history_feats
distance = ||z_real - z_pred||
```

6. All samples saved to:

```
data/corrector_data_<model_id>.pt
```

---

# 6. Corrector Training

Script: `scripts/train_corrector.py`

Dataset fields:

- `z_real`
- `z_pred`
- `a_plan`
- `a_teacher`
- `history_feats`

Loss:

```
L = MSE(a_corr, a_teacher) + λ * ||a_corr - a_plan||²
```

Each trained model is saved as:

```
correctors/corrector_<model_id>_<type>.pth
```

Types:

- `two_tower`
- `temporal`

---

# 7. Pretrained Checkpoint Handling

Implemented in `tdmpc2/utils_ckpt.py`.

### **Listing**

```
list_pretrained_checkpoints() → {model_id: path}
```

### **Loading**

```
load_pretrained_tdmpc2(model_id, checkpoint_path, task, device, plan_horizon)
```

This:

- Instantiates the agent with the correct architecture
- Loads weights
- Normalizes the task name (`humanoid-run` → `humanoid_run`)
- Returns `(agent, cfg)`

---

# 8. New Multi-Step Evaluation Script

File: `scripts/eval_mt30_humanoid.py`

**This script is fully standalone**  
(does NOT use `eval_corrector.py`).

### Evaluates:

- mt30-5M
- mt30-19M
- mt30-48M
- mt30-317M

on `humanoid-run` with:

```
baseline
2_step    naive / two_tower / temporal
3_step    naive / two_tower / temporal
5_step    naive / two_tower / temporal
6_step    naive / two_tower / temporal
```

### Produces:

- A CSV with all results
- A matplotlib plot of mean return

---

# 9. Usage Examples

### Collect data

```
python scripts/collect_corrector_data.py \
  --task humanoid-run \
  --model_id mt30-5M \
  --checkpoint_dir tdmpc2_pretrained \
  --episodes 20 \
  --output data/corrector_data_mt30-5M.pt
```

### Train correctors

```
python scripts/train_corrector.py \
  --model_id mt30-5M \
  --corrector_type both
```

### Evaluate

```
python scripts/eval_mt30_humanoid.py \
  --checkpoint_dir tdmpc2_pretrained \
  --corrector_dir correctors \
  --task humanoid-run \
  --episodes 10
```

---

# 10. Invariants the Codebase Must Maintain

- **Model IDs must be preserved** exactly as checkpoint stems.
- **Task names** must be normalized internally but CLI input may include `-`.
- **Corrector inputs** must match dataset format exactly.
- **Speculative execution** must preserve the block semantics:
  - 1 planner call per block
  - No replanning inside the block
  - Extended latent rollouts for steps >3
- **eval_mt30_humanoid.py must remain standalone** unless explicitly changed.

---

# 11. Summary

This document captures the full implementation state of speculative TD-MPC2 execution with corrector distillation for the `sc-tdmpc2` repository.

It defines:

- Multi-step speculative inference
- Corrector architectures
- Data collection
- Training pipeline
- Evaluation procedure
- File naming and model ID conventions
- Repository structure
- Expected interfaces and invariants

This file should be kept up-to-date as the system evolves.