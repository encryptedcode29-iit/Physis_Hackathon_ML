import argparse
import os
import numpy as np
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from quantum_env import QuantumOpticsEnv


class FidelityTracker(BaseCallback):
    def __init__(self, log_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.best_fidelity = 0.0
        self.episode_fidelities = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info and "fidelity" in info:
                fid = info["fidelity"]
                self.episode_fidelities.append(fid)
                if fid > self.best_fidelity:
                    self.best_fidelity = fid

        if self.num_timesteps % self.log_freq == 0 and self.verbose:
            recent = self.episode_fidelities[-100:] if self.episode_fidelities else [0]
            avg_fid = np.mean(recent)
            max_fid = max(recent) if recent else 0
            print(
                f"  [Step {self.num_timesteps:>8d}]  "
                f"Avg fidelity (last 100): {avg_fid:.4f}  |  "
                f"Best ever: {self.best_fidelity:.4f}  |  "
                f"Recent max: {max_fid:.4f}"
            )
        return True

    def _on_training_end(self):
        if self.verbose:
            print(f"\n[RESULT] Training complete. Best fidelity: {self.best_fidelity:.6f}")


class EarlyStopOnConvergence(BaseCallback):
    def __init__(self, threshold: float = 0.99, patience: int = 5,
                 log_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.threshold   = threshold
        self.patience    = patience
        self.log_freq    = log_freq
        self.consecutive = 0
        self.episode_fidelities = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info and "fidelity" in info:
                self.episode_fidelities.append(info["fidelity"])

        if self.num_timesteps % self.log_freq == 0 and self.episode_fidelities:
            recent_avg = np.mean(self.episode_fidelities[-100:])
            if recent_avg >= self.threshold:
                self.consecutive += 1
                if self.verbose:
                    print(f"  [EarlyStop] Avg fidelity {recent_avg:.4f} >= {self.threshold} "
                          f"({self.consecutive}/{self.patience})")
                if self.consecutive >= self.patience:
                    print(f"\n  [EarlyStop] CONVERGED — stopping at step {self.num_timesteps}")
                    return False   
            else:
                self.consecutive = 0  
        return True


def make_env(
    target: str = "ghz",
    num_qubits: int = int(os.getenv("NUM_QUBITS", 4)),
    max_steps: Optional[int] = None,
    reward_shaping: bool = True,
    step_penalty: float = 0.01,
    log_dir: str = "./logs",
) -> Monitor:
    os.makedirs(log_dir, exist_ok=True)
    env = QuantumOpticsEnv(
        target_state_name=target,
        num_qubits=num_qubits,
        max_steps=max_steps,
        reward_shaping=reward_shaping,
        step_penalty=step_penalty,
    )
    env = Monitor(env, filename=os.path.join(log_dir, "monitor"))
    return env


def train(args):
    nq = getattr(args, "num_qubits", int(os.getenv("NUM_QUBITS", 4)))
    print("=" * 60)
    print("  Quantum Optical RL — PPO Training")
    print("=" * 60)
    print(f"  Target state:    {args.target.upper()}")
    print(f"  Num qubits:      {nq}")
    print(f"  Max steps:       {args.max_steps if args.max_steps else 'auto'}")
    print(f"  Total timesteps: {args.timesteps:,}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Save dir:        {args.save_dir}")
    print("=" * 60 + "\n")

    env = DummyVecEnv([
        lambda nq=nq, t=args.target, ms=args.max_steps, ld=args.log_dir: make_env(
            target=t,
            num_qubits=nq,
            max_steps=ms,
            log_dir=ld,
        )
    ])

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=4096,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         
        vf_coef=0.3,
        max_grad_norm=0.3,
        verbose=1,
        tensorboard_log=None,
        seed=args.seed,
    )

    fidelity_tracker = FidelityTracker(log_freq=args.log_freq)
    early_stop = EarlyStopOnConvergence(
        threshold=0.99,
        patience=5,
        log_freq=args.log_freq,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.timesteps // 10, 1000),
        save_path=args.save_dir,
        name_prefix="quantum_ppo",
    )

    eval_env = DummyVecEnv([
        lambda nq=nq, t=args.target, ms=args.max_steps, ld=os.path.join(args.log_dir, "eval"): make_env(
            target=t,
            num_qubits=nq,
            max_steps=ms,
            log_dir=ld,
        )
    ])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.log_dir,
        eval_freq=max(args.timesteps // 5, 5000),
        n_eval_episodes=5,
        deterministic=True,
    )

    print("[START] Starting training...\n")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[fidelity_tracker, early_stop, checkpoint_cb, eval_cb],
        progress_bar=False,
    )

    final_path = os.path.join(args.save_dir, "quantum_ppo_final")
    model.save(final_path)
    print(f"\n[SAVED] Final model saved to: {final_path}.zip")

    print("\n" + "=" * 60)
    print("  Final Evaluation (10 episodes)")
    print("=" * 60)
    evaluate(model, args.target, nq, args.max_steps)

    env.close()
    eval_env.close()


def evaluate(model, target: str = "ghz",
             num_qubits: int = int(os.getenv("NUM_QUBITS", 4)),
             max_steps: Optional[int] = None,
             n_episodes: int = 10):
    env = QuantumOpticsEnv(
        target_state_name=target,
        num_qubits=num_qubits,
        max_steps=max_steps,
        render_mode="human",
    )

    fidelities = []
    best_fidelity = 0
    best_circuit = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            done = terminated or truncated

        fid = info["fidelity"]
        fidelities.append(fid)

        if fid > best_fidelity:
            best_fidelity = fid
            best_circuit = info["circuit"]

        print(f"  Episode {ep + 1:2d}: fidelity={fid:.6f}, "
              f"components={info['num_components']}, reward={episode_reward:.4f}")

    print(f"\n[STATS] Mean fidelity:  {np.mean(fidelities):.6f}")
    print(f"        Best fidelity:  {best_fidelity:.6f}")
    print(f"\n[BEST] Best circuit discovered:")
    for i, comp_str in enumerate(best_circuit):
        print(f"     {i + 1}. {comp_str}")

    env.close()
    return best_fidelity


def load_and_evaluate(args):
    """Load a saved model and run evaluation."""
    print(f"Loading model from: {args.model_path}")
    model = PPO.load(args.model_path)
    evaluate(model, args.target,
             getattr(args, "num_qubits", int(os.getenv("NUM_QUBITS", 4))),
             args.max_steps, n_episodes=args.eval_episodes)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for quantum optical experiment design."
    )
    parser.add_argument(
        "--target", type=str, default="ghz",
        help="Target state: preset name (ghz/w/bell), file path (.npy/.txt/.csv), or inline vector"
    )
    parser.add_argument(
        "--num_qubits", type=int,
        default=int(os.getenv("NUM_QUBITS", 4)),
        choices=[1, 2, 3, 4],
        help="Number of qubits (1-4). Also reads NUM_QUBITS env var. Default: 4"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Max components per episode. Default: auto-scaled (1->5, 2->8, 3->12, 4->15)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="Total training timesteps (default: 200000)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Mini-batch size (default: 64)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./models",
        help="Directory to save models (default: ./models)"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs",
        help="Directory for training logs (default: ./logs)"
    )
    parser.add_argument(
        "--log_freq", type=int, default=5000,
        help="Print fidelity stats every N steps (default: 5000)"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate a saved model instead of training"
    )
    parser.add_argument(
        "--model_path", type=str, default="./models/best_model.zip",
        help="Path to saved model for evaluation (default: ./models/best_model.zip)"
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.evaluate:
        load_and_evaluate(args)
    else:
        train(args)