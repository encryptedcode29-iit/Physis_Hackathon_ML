import argparse
import os
from quantum_env import QuantumOpticsEnv


def demo_random_agent(target: str = "ghz", num_qubits: int = 4,
                      max_steps=None, episodes: int = 5):
    print("=" * 60)
    print(f"  Quantum Optics RL — Random Agent Demo ({num_qubits}q)")
    print("=" * 60)

    env = QuantumOpticsEnv(
        target_state_name=target,
        num_qubits=num_qubits,
        max_steps=max_steps,
        render_mode="human",
    )
    print(f"  max_steps = {env.max_steps}  |  actions = {env.num_actions}\n")

    best_fid = 0
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0

        print(f"\n--- Episode {ep + 1} ---")
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        env.render()
        fid = info["fidelity"]
        best_fid = max(best_fid, fid)
        print(f"  Reward: {ep_reward:.4f}  |  Fidelity: {fid:.6f}")

    print(f"\n[BEST] Best fidelity from random agent: {best_fid:.6f}")
    env.close()


def demo_known_circuit(target: str = "ghz", num_qubits: int = 4):
    print("=" * 60)
    print(f"  Known Optimal GHZ Circuit ({num_qubits}q)")
    print("=" * 60)

    env = QuantumOpticsEnv(
        target_state_name=target,
        num_qubits=num_qubits,
        render_mode="human",
    )
    obs, info = env.reset()

    catalogue = env.action_catalogue
    action_map = {}

    for i, comp in enumerate(catalogue):
        if comp.name == "Hadamard" and comp.qubits == (0,):
            action_map["H(0)"] = i
        for q in range(num_qubits - 1):
            if comp.name == "CNOT" and comp.qubits == (q, q + 1):
                action_map[f"CNOT({q},{q+1})"] = i

    ordered_keys = ["H(0)"] + [f"CNOT({q},{q+1})" for q in range(num_qubits - 1)]
    actions_to_take = [(key, action_map[key]) for key in ordered_keys if key in action_map]

    print("\nApplying: " + " -> ".join(k for k, _ in actions_to_take) + "\n")

    for name, action_id in actions_to_take:
        obs, reward, terminated, truncated, info = env.step(action_id)
        print(f"  Applied {name}: fidelity = {info['fidelity']:.6f}, reward = {reward:.4f}")

    print(f"\n  Final fidelity: {info['fidelity']:.6f}")
    env.render()
    env.close()


def train(target: str, num_qubits: int, timesteps: int, max_steps=None):
    from train_ppo import train as ppo_train

    class Args:
        pass

    args = Args()
    args.target     = target
    args.num_qubits = num_qubits
    args.max_steps  = max_steps   
    args.timesteps  = timesteps
    args.lr         = 3e-4
    args.batch_size = 64
    args.seed       = 42
    args.save_dir   = "./models"
    args.log_dir    = "./logs"
    args.log_freq   = 5000

    ppo_train(args)


def main():
    parser = argparse.ArgumentParser(description="Quantum Optical RL Design Engine")
    parser.add_argument("--mode", type=str, default="demo",
                        choices=["train", "demo", "eval", "known"],
                        help="Mode: train / demo / eval / known")
    parser.add_argument("--target", type=str, default="ghz",
                        help="Target: preset (ghz/w/bell), file path, or inline vector")
    parser.add_argument("--num_qubits", type=int,
                        default=int(os.getenv("NUM_QUBITS", 4)),
                        choices=[1, 2, 3, 4],
                        help="Number of qubits (1-4). Also reads NUM_QUBITS env var.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max components per episode. Default: auto-scaled per num_qubits "
                             "(1→5, 2→8, 3→12, 4→15)")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model",     type=str, default="./models/best_model.zip")
    parser.add_argument("--episodes",  type=int, default=5)

    args = parser.parse_args()

    if args.mode == "demo":
        demo_random_agent(args.target, args.num_qubits, args.max_steps, args.episodes)
    elif args.mode == "known":
        demo_known_circuit(args.target, args.num_qubits)
    elif args.mode == "train":
        train(args.target, args.num_qubits, args.timesteps, args.max_steps)
    elif args.mode == "eval":
        from stable_baselines3 import PPO as PPOModel
        from train_ppo import evaluate
        model = PPOModel.load(args.model)
        evaluate(model, args.target, args.num_qubits, args.max_steps, args.episodes)


if __name__ == "__main__":
    main()