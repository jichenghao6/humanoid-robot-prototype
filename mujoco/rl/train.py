"""RSL-RL PPO training script for humanoid_robot Walk & Run on RTX 4070 (MuJoCo-Warp)."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

# Ensure project root / mujoco/rl is importable regardless of cwd
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from rsl_rl.runners import OnPolicyRunner
import torch

from humanoid_env import EnvConfig, HumanoidLocomotionEnv


def build_train_cfg(args: argparse.Namespace) -> dict:
    return {
        "seed": args.seed,
        "num_steps_per_env": 24,
        "save_interval": args.save_interval,
        "logger": "tensorboard",
        "check_for_nan": True,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["critic"],
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.65 if args.resume else 0.85,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.004 if args.task == "run" else 0.006,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.012,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train humanoid_robot Walk/Run with MuJoCo-Warp + RSL-RL")
    parser.add_argument("--task", choices=["walk", "run"], default="walk")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to continue/curriculum-train from")
    parser.add_argument("--logdir", type=str, default=str(THIS_DIR / "logs"))
    parser.add_argument("--no_pushes", action="store_true")
    parser.add_argument(
        "--update-checkpoints",
        action="store_true",
        help="Also overwrite tracked mujoco/rl/checkpoints/latest_{task}.{pt,onnx} after training",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = EnvConfig(
        task=args.task,
        num_envs=args.num_envs,
        enable_pushes=not args.no_pushes,
    )
    env = HumanoidLocomotionEnv(env_cfg, device=device)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.logdir) / f"{args.task}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = build_train_cfg(args)
    runner = OnPolicyRunner(env, train_cfg, log_dir=str(run_dir), device=device)

    if args.resume:
        print(f"Loading checkpoint for curriculum/resume: {args.resume}")
        runner.load(
            args.resume,
            load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False},
        )

    print(f"Starting training: task={args.task}, num_envs={args.num_envs}, iterations={args.max_iterations}, device={device}")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    # Export latest checkpoint copy for easy play.py lookup
    final_ckpt = run_dir / f"model_{runner.current_learning_iteration}.pt"
    ckpt_bytes = final_ckpt.read_bytes()
    latest_ckpt = Path(args.logdir) / f"latest_{args.task}.pt"
    latest_ckpt.write_bytes(ckpt_bytes)
    curated_dir = THIS_DIR / "checkpoints"
    if args.update_checkpoints:
        curated_dir.mkdir(parents=True, exist_ok=True)
        (curated_dir / f"latest_{args.task}.pt").write_bytes(ckpt_bytes)
        print(f"Updated curated checkpoint: {curated_dir / f'latest_{args.task}.pt'}")
    print(f"Training complete! Final checkpoint: {final_ckpt}")
    print(f"Saved convenience copy: {latest_ckpt}")

    # Export actor ONNX for portable inference (taking flat 74-dim policy tensor)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        policy_actor = runner.get_inference_policy(device="cpu")

        class _FlatPolicyExport(torch.nn.Module):
            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, policy_obs: torch.Tensor) -> torch.Tensor:
                from tensordict import TensorDict
                return self.model(TensorDict({"policy": policy_obs}, batch_size=[policy_obs.shape[0]]))

        flat_wrapper = _FlatPolicyExport(policy_actor).eval()
        dummy_flat = env.get_observations()["policy"][:1].to("cpu")
        onnx_path = Path(args.logdir) / f"latest_{args.task}.onnx"
        torch.onnx.export(
            flat_wrapper,
            (dummy_flat,),
            str(onnx_path),
            input_names=["obs"],
            output_names=["actions"],
            opset_version=18,
            dynamo=False,
        )
        if args.update_checkpoints:
            curated_dir.mkdir(parents=True, exist_ok=True)
            (curated_dir / f"latest_{args.task}.onnx").write_bytes(onnx_path.read_bytes())
            print(f"Updated curated ONNX: {curated_dir / f'latest_{args.task}.onnx'}")
        print(f"Exported ONNX policy: {onnx_path}")
    except Exception as exc:
        print(f"Optional ONNX export skipped: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
