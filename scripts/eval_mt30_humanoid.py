import argparse
import sys
import math
import statistics
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mt30 TD-MPC2 teachers with correctors on humanoid-run")
    parser.add_argument("--checkpoint_dir", type=str, default="tdmpc2_pretrained")
    parser.add_argument("--corrector_dir", type=str, default="correctors")
    parser.add_argument("--task", type=str, default="humanoid-run")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
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


def run_episode_baseline(env, agent, max_steps: int) -> float:
    """Replan every step with TD-MPC2 (teacher only)."""

    obs = env.reset()
    ep_ret = 0.0
    for t in range(max_steps):
        obs_tensor = ensure_tensor(obs)
        with torch.no_grad():
            action = agent.act(obs_tensor, step=t, eval_mode=True)
        if isinstance(action, torch.Tensor):
            action_env = action.cpu()
        else:
            action_env = torch.tensor(action)
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
) -> float:
    """
    Multi-step execution from a TD-MPC2 3-step plan, optionally using a corrector.
    """

    obs = env.reset()
    ep_ret = 0.0
    t = 0
    mismatch_history: Deque[torch.Tensor] = deque(maxlen=history_len)

    while t < max_steps:
        obs_tensor = ensure_tensor(obs, device=agent.device)
        with torch.no_grad():
            a_plan_seq, z_pred_seq = agent.plan_with_predicted_latents(
                obs_tensor, eval_mode=True, horizon=3
            )
        # Convert planner outputs to lists for easier indexing
        plan_actions: List[torch.Tensor] = [a for a in a_plan_seq]
        pred_latents: List[torch.Tensor] = [z for z in z_pred_seq]
        last_action: Optional[torch.Tensor] = None

        for k in range(exec_horizon):
            if t >= max_steps:
                break

            obs_tensor = ensure_tensor(obs, device=agent.device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            with torch.no_grad():
                z_real = agent.model.encode(obs_tensor, None).squeeze(0)

            if k < len(pred_latents):
                z_pred_k = pred_latents[k].to(agent.device)
            else:
                if last_action is None:
                    last_action = plan_actions[-1].to(agent.device)
                next_z = agent.model.next(pred_latents[-1].unsqueeze(0).to(agent.device), last_action.unsqueeze(0), None)
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

            obs, rew, done, _ = env.step(action.detach().cpu())
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

    ckpt_root = Path(args.checkpoint_dir)
    corr_root = Path(args.corrector_dir)

    mt30_ckpts = sorted(p for p in ckpt_root.glob("mt30-*.pt") if p.is_file())
    if not mt30_ckpts:
        print(f"No mt30-* checkpoints found in {ckpt_root}")
        sys.exit(1)

    results = []

    from tdmpc2.utils_ckpt import load_pretrained_tdmpc2
    from tdmpc2.envs import make_env
    from tdmpc2.common.parser import populate_env_dims, normalize_task_name
    from tdmpc2.corrector import TwoTowerCorrector, TemporalTransformerCorrector

    exec_horizons = [2, 3, 5, 6]

    for ckpt in mt30_ckpts:
        model_id = ckpt.stem
        agent, cfg = load_pretrained_tdmpc2(
            model_id=model_id,
            checkpoint_path=str(ckpt),
            task=args.task,
            device=args.device,
            plan_horizon=args.plan_horizon,
        )

        cfg.task = normalize_task_name(args.task)
        cfg_env, _ = populate_env_dims(cfg)
        env = make_env(cfg_env)

        two_tower = None
        temporal = None

        two_tower_ckpt = corr_root / f"corrector_{model_id}_two_tower.pth"
        temporal_ckpt = corr_root / f"corrector_{model_id}_temporal.pth"

        if two_tower_ckpt.is_file():
            ckpt_tt = torch.load(two_tower_ckpt, map_location=args.device)
            latent_dim = ckpt_tt.get("latent_dim", agent.latent_dim)
            act_dim = ckpt_tt.get("act_dim", agent.act_dim)
            two_tower = TwoTowerCorrector(
                latent_dim=latent_dim,
                act_dim=act_dim,
                **ckpt_tt.get("hparams", {}),
            ).to(args.device)
            two_tower.load_state_dict(ckpt_tt["state_dict"])
            two_tower.eval()

        if temporal_ckpt.is_file():
            ckpt_tmp = torch.load(temporal_ckpt, map_location=args.device)
            latent_dim = ckpt_tmp.get("latent_dim", agent.latent_dim)
            act_dim = ckpt_tmp.get("act_dim", agent.act_dim)
            temporal = TemporalTransformerCorrector(
                latent_dim=latent_dim,
                act_dim=act_dim,
                **ckpt_tmp.get("hparams", {}),
            ).to(args.device)
            temporal.load_state_dict(ckpt_tmp["state_dict"])
            temporal.eval()

        modes = ["baseline"]
        for horizon in exec_horizons:
            for suffix in ["naive", "two_tower", "temporal"]:
                modes.append(f"{horizon}_step_{suffix}")

        for mode in modes:
            returns: List[float] = []
            if mode == "baseline":
                exec_h = 1
                corr_type = "none"
            else:
                parts = mode.split("_")
                exec_h = int(parts[0])
                corr_suffix = parts[-1]
                if corr_suffix == "two" and len(parts) > 3:
                    corr_suffix = "two_tower"
                corr_map = {
                    "naive": None,
                    "two": two_tower,
                    "tower": two_tower,
                    "two_tower": two_tower,
                    "temporal": temporal,
                }
                corr = corr_map.get(corr_suffix, None)
                corr_type = "none" if corr is None else ("two_tower" if corr is two_tower else "temporal")

            for _ in range(args.episodes):
                if mode == "baseline":
                    ep_ret = run_episode_baseline(env, agent, args.max_steps)
                else:
                    ep_ret = run_episode_multistep(
                        env,
                        agent,
                        args.max_steps,
                        exec_horizon=exec_h,
                        corrector=corr,
                    )
                returns.append(ep_ret)

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

    fig, ax = plt.subplots(figsize=(9, 5))
    for horizon in sorted(df["exec_horizon"].unique()):
        sub_h = df[df["exec_horizon"] == horizon]
        for ctype in ["none", "two_tower", "temporal"]:
            sub = sub_h[sub_h["corrector_type"] == ctype]
            if sub.empty:
                continue
            label = f"{ctype}, {horizon}-step"
            ax.plot(sub["model_id"], sub["mean_return"], marker="o", label=label)
    ax.set_xlabel("model_id (mt30-*)")
    ax.set_ylabel("mean_return")
    ax.set_title(
        "mt30 TD-MPC2 on humanoid-run: multi-step exec (2/3/5/6) with/without correctors"
    )
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_path = Path(args.results_plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
