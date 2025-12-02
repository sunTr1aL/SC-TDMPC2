import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import math
import statistics
from collections import deque
from typing import Deque, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tdmpc2.utils_ckpt import load_pretrained_tdmpc2, list_pretrained_checkpoints
from tdmpc2.envs import make_env
from tdmpc2.common import TASK_SET
from tdmpc2.common.parser import populate_env_dims
from tdmpc2.corrector import TwoTowerCorrector, TemporalTransformerCorrector

def model_size_key(model_id):
    parts = model_id.split("-")
    if len(parts) < 2: return 0
    size_str = parts[-1].replace("M", "")
    if size_str.isdigit():
        return int(size_str)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mt30 TD-MPC2 teachers with correctors on humanoid-run")
    parser.add_argument("--checkpoint_dir", type=str, default="tdmpc2_pretrained")
    parser.add_argument("--corrector_dir", type=str, default="correctors")
    parser.add_argument("--task", type=str, default="walker-run", help="Task name")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_size", type=str, default=None, help="Filter by model size")
    parser.add_argument("--corrector_hidden_dim", type=int, default=512)
    parser.add_argument("--corrector_layers", type=int, default=4)
    parser.add_argument("--plan_horizon", type=int, default=3)
    parser.add_argument(
        "--results_csv",
        type=str,
        default="results/corrector_eval/mt30_humanoid_small_eval.csv",
    )
    parser.add_argument(
        "--results_plot",
        type=str,
        default="results/corrector_eval/mt30_humanoid_small_eval.png",
    )
    return parser.parse_args()


def ensure_tensor(x, device: Optional[str] = None) -> torch.Tensor:
    tensor = torch.as_tensor(x, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def run_episode_baseline(env, agent, max_steps: int, task_idx=None) -> float:
    """Replan every step with TD-MPC2 (teacher only)."""

    obs = env.reset()
    ep_ret = 0.0
    for t in range(max_steps):
        obs_tensor = ensure_tensor(obs)
        # Truncate obs to match agent's expected dim
        expected_obs_dim = agent.cfg.encoder_in_dim - agent.cfg.task_emb_dim
        if obs_tensor.shape[-1] > expected_obs_dim:
            obs_tensor = obs_tensor[..., :expected_obs_dim]
            
        with torch.no_grad():
            action = agent.act(obs_tensor, t0=(t==0), eval_mode=True, task=task_idx)
        if isinstance(action, torch.Tensor):
            action_env = action.cpu()
        else:
            action_env = action
        
        if action_env.shape[-1] < env.action_space.shape[0]:
            padding = torch.zeros(action_env.shape[:-1] + (env.action_space.shape[0] - action_env.shape[-1],), dtype=action_env.dtype)
            action_env = torch.cat([action_env, padding], dim=-1)

        obs, rew, done, _ = env.step(action_env)
        ep_ret += float(rew)
        if done:
            break
    return ep_ret


def _stack_history(history: Deque[torch.Tensor], history_len: int, device: str) -> Optional[torch.Tensor]:
    if not history:
        return None
    feats = list(history)
    if len(feats) < history_len:
        pad = [torch.zeros_like(feats[0]) for _ in range(history_len - len(feats))]
        feats = pad + feats
    feats = feats[-history_len:]
    return torch.stack(feats, dim=0).unsqueeze(0).to(device)


def run_episode_multistep(
    env,
    agent,
    max_steps: int,
    exec_horizon: int,
    corrector=None,
    history_len: int = 4,
    task_idx=None,
) -> float:
    """
    Multi-step execution from a TD-MPC2 3-step plan, optionally using a corrector.
    """

    if task_idx is None:
        obs = env.reset()
    else:
        try:
            obs = env.reset(task_idx=task_idx)
        except TypeError:
            obs = env.reset()

    ep_ret = 0.0
    t = 0
    mismatch_history: Deque[torch.Tensor] = deque(maxlen=history_len)

    while t < max_steps:
        obs_tensor = ensure_tensor(obs, device=agent.device)
        # Truncate obs to match agent's expected dim
        expected_obs_dim = agent.cfg.encoder_in_dim - agent.cfg.task_emb_dim
        if obs_tensor.shape[-1] > expected_obs_dim:
            obs_tensor = obs_tensor[..., :expected_obs_dim]

        with torch.no_grad():
            a_plan_seq, z_pred_seq = agent.plan_with_predicted_latents(
                obs_tensor, eval_mode=True, horizon=3, task=task_idx
            )
        # Convert planner outputs to lists for easier indexing
        plan_actions: List[torch.Tensor] = [a for a in a_plan_seq]
        pred_latents: List[torch.Tensor] = [z for z in z_pred_seq]
        last_action: Optional[torch.Tensor] = None

        for k in range(exec_horizon):
            if t >= max_steps:
                break

            obs_tensor = ensure_tensor(obs, device=agent.device)
            if obs_tensor.shape[-1] > expected_obs_dim:
                obs_tensor = obs_tensor[..., :expected_obs_dim]
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            with torch.no_grad():
                z_real = agent.model.encode(obs_tensor, task_idx).squeeze(0)

            if k < len(pred_latents):
                z_pred_k = pred_latents[k].to(agent.device)
            else:
                if last_action is None:
                    last_action = plan_actions[-1].to(agent.device)
                z_in = pred_latents[-1].unsqueeze(0).to(agent.device)
                if z_in.ndim != last_action.ndim:
                    print(f"[CRASH DEBUG] z_in shape: {z_in.shape}, last_action shape: {last_action.shape}")
                    if last_action.ndim == 1:
                        last_action = last_action.unsqueeze(0)
                next_z = agent.model.next(z_in, last_action, task_idx)
                z_pred_k = next_z.squeeze(0)
                pred_latents.append(z_pred_k.detach().cpu())

            a_plan_k = plan_actions[min(k, len(plan_actions) - 1)].to(agent.device)

            mismatch_hist_tensor = _stack_history(mismatch_history, history_len, agent.device)
            if corrector is None:
                action = a_plan_k
            else:
                action = corrector(
                    z_real,
                    z_pred_k,
                    a_plan_k,
                    mismatch_history=mismatch_hist_tensor,
                )

            feature = torch.cat(
                [
                    z_real.detach().cpu(),
                    z_pred_k.detach().cpu(),
                    (z_real - z_pred_k).detach().cpu(),
                    a_plan_k.detach().cpu(),
                ],
                dim=-1,
            )
            mismatch_history.append(feature)

            action_env = action.detach().cpu()
            if action_env.shape[-1] < env.action_space.shape[0]:
                padding = torch.zeros(action_env.shape[:-1] + (env.action_space.shape[0] - action_env.shape[-1],), dtype=action_env.dtype)
                action_env = torch.cat([action_env, padding], dim=-1)
            
            obs, rew, done, _ = env.step(action_env)
            ep_ret += float(rew)
            last_action = action.detach()
            t += 1
            if done:
                break
        if done:
            break
    return ep_ret


def main() -> None:
    args = parse_args()

    # ckpt_root = Path(args.checkpoint_dir) # This line is no longer needed as list_pretrained_checkpoints handles it
    corr_root = Path(args.corrector_dir)

    mt30_ckpts = list_pretrained_checkpoints(args.checkpoint_dir, model_size_filter=args.model_size)
    if not mt30_ckpts:
        print(f"No mt30-* checkpoints found in {args.checkpoint_dir} with filter {args.model_size}")
        sys.exit(1)

    results = []

    # Imports moved to top

    exec_horizons = [2, 3, 4, 5, 6]

    for model_id, info in mt30_ckpts.items():
        print(f"Evaluating model: {model_id}")
        ckpt_path = Path(info["path"])
        
        # Load agent
        try:
            agent, cfg = load_pretrained_tdmpc2(
                str(ckpt_path),
                device=args.device,
                # task=args.task, # Load as multitask (mt30)
                obs_type='state',
            )
        except Exception as e:
            print(f"Failed to load agent {model_id}: {e}")
            continue # Skip to the next model if loading fails
        original_action_dim = cfg.action_dim
        
        # Load full multitask environment to match training setup
        cfg.task = "mt30"
        cfg.multitask = True
        cfg_env, _ = populate_env_dims(cfg)
        env = make_env(cfg_env)
        
        # Identify task index for walker-run
        if args.task in TASK_SET['mt30']:
            task_idx = TASK_SET['mt30'].index(args.task)
        else:
            raise ValueError(f"Task {args.task} not found in mt30 set")
        
        print(f"[DEBUG] Evaluating on task {args.task} (Index {task_idx}) using MultitaskWrapper")

        two_tower = None
        temporal = None

        two_tower_ckpt = corr_root / f"corrector_{model_id}_two_tower.pth"
        temporal_ckpt = corr_root / f"corrector_{model_id}_temporal.pth"

        if two_tower_ckpt.is_file():
            ckpt_tt = torch.load(two_tower_ckpt, map_location=args.device)
            latent_dim = ckpt_tt.get("latent_dim", agent.cfg.latent_dim)
            act_dim = ckpt_tt.get("act_dim", agent.cfg.action_dim)
            # Use hparams from ckpt if available, else use args
            hparams = ckpt_tt.get("hparams", {})
            if "hidden_dim" not in hparams: hparams["hidden_dim"] = args.corrector_hidden_dim
            if "num_layers" not in hparams: hparams["num_layers"] = args.corrector_layers
            
            two_tower = TwoTowerCorrector(
                latent_dim=latent_dim,
                act_dim=act_dim,
                **hparams,
            ).to(args.device)
            two_tower.load_state_dict(ckpt_tt["corrector"])
            two_tower.eval()

        if temporal_ckpt.is_file():
            ckpt_tmp = torch.load(temporal_ckpt, map_location=args.device)
            latent_dim = ckpt_tmp.get("latent_dim", agent.cfg.latent_dim)
            act_dim = ckpt_tmp.get("act_dim", agent.cfg.action_dim)
            # Use hparams from ckpt if available, else use args
            hparams = ckpt_tmp.get("hparams", {})
            if "hidden_dim" not in hparams: hparams["hidden_dim"] = args.corrector_hidden_dim
            if "num_layers" not in hparams: hparams["num_layers"] = args.corrector_layers

            temporal = TemporalTransformerCorrector(
                latent_dim=latent_dim,
                act_dim=act_dim,
                **hparams,
            ).to(args.device)
            temporal.load_state_dict(ckpt_tmp["corrector"])
            temporal.eval()


        modes = ["baseline"]
        for h in exec_horizons:
            modes.append(f"{h}_none")
            if two_tower is not None:
                modes.append(f"{h}_two_tower")
            if temporal is not None:
                modes.append(f"{h}_temporal")
        
        # Remove 1_ modes if they are confusing, but user asked to keep them for now or debug them.
        # Actually, user said "The presence of '1_two_tower' ... suggests horizon=1 modes are using correctors... may be part of the bug."
        # But for now I will keep them to see the debug output.

        for mode in modes:
            print(f"Evaluating mode: {mode}")
            exec_h = 0
            corr = None
            corr_type = "none"

            if mode == "baseline":
                pass
            else:
                parts = mode.split("_")
                exec_h = int(parts[0])
                corr_suffix = "_".join(parts[1:]) # Handle two_tower
                
                corr_map = {
                    "two_tower": two_tower,
                    "temporal": temporal,
                }
                corr = corr_map.get(corr_suffix, None)
                corr_type = "none" if corr is None else ("two_tower" if corr is two_tower else "temporal")

            returns = []
            for i in range(args.episodes):
                if mode == "baseline":
                    ep_ret = run_episode_baseline(env, agent, args.max_steps, task_idx=task_idx)
                else:
                    ep_ret = run_episode_multistep(
                        env,
                        agent,
                        args.max_steps,
                        exec_horizon=exec_h,
                        corrector=corr,
                        task_idx=task_idx,
                    )
                returns.append(ep_ret)
                print(f"Episode {i}: {ep_ret}")

            mean_ret = float(np.mean(returns))
            median_ret = float(np.median(returns))
            std_ret = float(np.std(returns))
            min_ret = float(np.min(returns))

            results.append(
                {
                    "model_id": model_id,
                    "mode": mode,
                    "exec_horizon": exec_h,
                    "corrector_type": corr_type,
                    "mean_return": mean_ret,
                    "median_return": median_ret,
                    "std_return": std_ret,
                    "min_return": min_ret,
                    "episodes": args.episodes,
                }
            )

    results_dir = Path(args.results_csv).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(args.results_csv, index=False)
    print(f"Saved mt30 humanoid eval to {args.results_csv}")

    # Summary reporting
    unique_models = sorted(df["model_id"].unique(), key=model_size_key)
    for model_id in unique_models:
        sub = df[df["model_id"] == model_id]
        print(f"[SUMMARY] model_id={model_id}")
        for H in sorted(sub["exec_horizon"].unique()):
            for ctype in ["none", "two_tower", "temporal"]:
                d = sub[(sub["exec_horizon"] == H) & (sub["corrector_type"] == ctype)]
                if d.empty:
                    continue
                print(f"  H={H}, type={ctype}, mean_return={d['mean_return'].iloc[0]:.4f}")

    # Plotting
    for horizon in [2, 3, 4, 5, 6]:
        sub_h = df[df["exec_horizon"] == horizon]
        if sub_h.empty:
             continue
        
        fig, ax = plt.subplots(figsize=(9, 5))
        
        # Baseline (H=1, none)
        df_baseline = df[(df["exec_horizon"] == 1) & (df["corrector_type"] == "none")]
        if not df_baseline.empty:
             # Sort by model size
             df_baseline = df_baseline.set_index("model_id").reindex(unique_models).reset_index()
             ax.plot(df_baseline["model_id"], df_baseline["mean_return"], marker="x", linestyle="--", label="Baseline (H=1)", color="gray")

        # H-step variants
        for ctype, color in [("none", "green"), ("two_tower", "blue"), ("temporal", "orange")]:
            sub = sub_h[sub_h["corrector_type"] == ctype]
            if sub.empty:
                continue
            sub = sub.set_index("model_id").reindex(unique_models).reset_index()
            if ctype == "none":
                label = f"No Corrector, {horizon}-step"
            else:
                label = f"{ctype}, {horizon}-step"
            ax.plot(sub["model_id"], sub["mean_return"], marker="o", label=label, color=color)
            
        ax.set_xlabel("Model Size")
        ax.set_ylabel("Mean Return")
        ax.set_title(f"TD-MPC2 Humanoid-Run: Horizon {horizon}")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        plot_path = Path(args.results_plot).parent / f"mt30_humanoid_eval_H{horizon}.png"
        fig.savefig(plot_path)
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
