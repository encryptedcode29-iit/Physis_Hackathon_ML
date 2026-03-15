# 🔬 Quantum Optical RL Design Engine

> **Physis Techne Symposium 2026 — ML Challenge**  
> *Automated design of quantum optical experiments using Reinforcement Learning*

---

## What is this?

Designing a quantum optical experiment is genuinely hard. You have a target quantum state you want to prepare — something exotic like a GHZ state or a custom entangled state — and you need to figure out which optical components to use, in what order, to actually produce it. Normally this requires deep domain expertise and a lot of intuition.

This project asks a different question: **can a machine figure it out on its own?**

The answer is yes — using Reinforcement Learning. We frame circuit design as a sequential decision problem: the agent picks optical components one at a time (beam splitters, wave plates, SPDC sources, detectors…), drops them onto a virtual optical table, and gets rewarded based on how close the resulting quantum state is to the target. Over many thousands of episodes, it learns which sequences of components work for which states.

No labelled dataset. No hand-crafted circuit templates. Just physics-guided trial and error.

---

## Quick Start

```bash
# 1. Clone and install
git clone <your-repo-url>
cd quantum_rl_fixed_copy
pip install -r requirements.txt

# 2. Train the agent
python train_ppo.py --target ghz --num_qubits 4 --timesteps 200000

# 3. Evaluate a saved model
python train_ppo.py --evaluate --model_path ./models/best_model.zip --num_qubits 4

# 4. Or use main.py for demo / known-circuit modes
python main.py --mode known --target ghz --num_qubits 4
python main.py --mode demo  --target ghz --num_qubits 4 --episodes 5
```

---

## Training by Qubit Count

The environment supports 1 to 4 qubits. Pass `--num_qubits` directly to `train_ppo.py`, or set the `NUM_QUBITS` environment variable. Both work.

### Linux / macOS

```bash
# 4-qubit GHZ (default)
python train_ppo.py --target ghz --num_qubits 4 --timesteps 200000

# 3-qubit GHZ
python train_ppo.py --target ghz --num_qubits 3 --timesteps 200000

# 2-qubit Bell state
python train_ppo.py --target bell --num_qubits 2 --timesteps 100000

# 1-qubit sanity check
python train_ppo.py --target ghz --num_qubits 1 --timesteps 50000

# Using the environment variable
NUM_QUBITS=3 python train_ppo.py --target ghz --timesteps 200000
```

### Windows (PowerShell)

```powershell
# 3-qubit GHZ via environment variable
$env:NUM_QUBITS = "3"
python train_ppo.py --target ghz --timesteps 200000

# Or pass the flag directly — cleaner and explicit
python train_ppo.py --target ghz --num_qubits 3 --timesteps 200000

# 4-qubit W state
python train_ppo.py --target w --num_qubits 4 --timesteps 200000

# Evaluate after training
python train_ppo.py --evaluate --model_path ./models/best_model.zip --num_qubits 3
```

### Windows (Command Prompt)

```cmd
:: 3-qubit GHZ via environment variable
set NUM_QUBITS=3
python train_ppo.py --target ghz --timesteps 200000

:: Or pass the flag directly
python train_ppo.py --target ghz --num_qubits 3 --timesteps 200000
```

---

## How to Specify a Target State

The system is not limited to preset states — you can hand it any quantum state you want to prepare.

| Method | Example | Notes |
|---|---|---|
| **Preset name** | `--target ghz` | Supports `ghz`, `w`, `bell` |
| **NumPy file** | `--target my_state.npy` | Complex128 array, length = 2^n |
| **Text file** | `--target my_state.txt` | One complex number per line |
| **Inline vector** | `--target "0.707 0 0 ... 0.707"` | Space or comma-separated |

### Built-in Preset States

| Name | Quantum State | Qubits |
|---|---|---|
| `ghz` | (1/√2)(&#124;000⟩ + &#124;111⟩) for 3q, (1/√2)(&#124;0000⟩ + &#124;1111⟩) for 4q | 1–4 |
| `w` | (1/2)(&#124;0001⟩ + &#124;0010⟩ + &#124;0100⟩ + &#124;1000⟩) | 4 |
| `bell` | (1/√2)(&#124;00⟩ + &#124;11⟩) | 2 |

---

## How It Works

The core idea: turn circuit design into a sequential decision problem and solve it with Reinforcement Learning.

```
Target state (any n-qubit state, n = 1..4)
            │
            ▼
  ┌──────────────────────────────────────┐
  │   PPO Policy Network                 │
  │                                      │
  │   Observes: current quantum state,   │
  │   target state, fidelity so far,     │
  │   circuit history                    │
  │                                      │
  │   Step 1 → pick Hadamard(q0)         │
  │   Step 2 → pick CNOT(0,1)            │
  │   Step 3 → pick CNOT(1,2)            │
  │   ...                                │
  └──────────────────────────────────────┘
            │
            ▼
  Physics simulator applies each component
  to the evolving quantum state vector
            │
            ▼
  Fidelity F(ρ_target, ρ_out) computed after each step
            │
            ▼
  Reward = incremental fidelity gain − step penalty
  Terminal bonus for high final fidelity
            │
            ▼
  PPO updates policy (early stop if converged)
            │
            ▼
  Output: discovered circuit = your photonic experiment
```

The fidelity metric used is the standard quantum fidelity:

```
F(ρ_target, ρ_out) = Tr[ √(√ρ_target · ρ_out · √ρ_target) ]²
```

For pure states this simplifies to |⟨ψ_target|ψ_out⟩|².

---

## Project Structure

```
quantum_rl_fixed_copy/
├── quantum_physics.py      # Core physics engine
├── quantum_components.py   # Optical component library + action catalogue
├── quantum_env.py          # Gymnasium RL environment
├── train_ppo.py            # PPO training + evaluation CLI
├── main.py                 # Entry point (train / demo / eval / known modes)
├── requirements.txt        # Python dependencies
├── models/                 # Saved checkpoints
│   ├── best_model.zip      # Best model by eval return
│   └── quantum_ppo_*_steps.zip
└── logs/                   # Training logs
    ├── monitor.monitor.csv
    └── evaluations.npz
```

### File Descriptions

**`quantum_physics.py`** — The physics simulator. Handles state vector representation (up to 16-dimensional for 4-qubit systems), gate construction via tensor products (2×2 up to 16×16 matrices), state evolution under unitary and non-unitary operations, and fidelity computation. Also builds preset target states like GHZ and W from scratch.

**`quantum_components.py`** — The optical component library. Every component (wave plate, beam splitter, SPDC source, detector, etc.) is a `QuantumComponent` dataclass carrying its gate matrix, target qubits, and parameters. `build_action_catalogue()` assembles all valid placements into a discrete action space. Resource tracking (max SPDC count, max spatial modes) enforces the problem constraints.

**`quantum_env.py`** — The Gymnasium environment. Wraps physics and components into a standard RL interface. Manages observations (quantum state, target state, fidelity, circuit history), reward computation, episode termination, and constraint enforcement. Observation space is a dict of real and imaginary state vector parts plus circuit bookkeeping.

**`train_ppo.py`** — Full PPO training pipeline using Stable-Baselines3. Includes a custom `FidelityTracker` callback (logs fidelity statistics during training), an `EarlyStopOnConvergence` callback (halts when avg fidelity stays above 0.99 for 5 consecutive log checks), checkpoint saving every 10% of training, and an evaluation loop that prints the best discovered circuit.

**`main.py`** — Convenience entry point. Four modes: `train` (delegates to train_ppo), `demo` (random agent baseline to show how hard the problem is), `eval` (load and test a saved model), `known` (run the hardcoded optimal GHZ circuit to verify the physics engine).

---

## The Optical Component Library

These are the building blocks the agent selects from. The library mirrors components used in real photonic quantum optics experiments.

### (a) Phase Shifter
Applies a phase e^{iφ} to a single qubit. Controls the relative phase between quantum states, enabling precise tuning of interference. Essential for constructing arbitrary single-qubit rotations when combined with wave plates. *Acts on 1 qubit.*

### (b) Half-Wave Plate (HWP)
Rotates the polarization of a photon by 2θ about the optical axis. An HWP at 22.5° is equivalent to a Hadamard gate in the polarization basis. The workhorse of any polarization-based experiment. *Acts on 1 qubit.*

### (c) Quarter-Wave Plate (QWP)
Introduces a π/2 phase shift between the two orthogonal polarization components, converting between linear and circular polarization. Together with HWP, the QWP+HWP combination can implement any single-qubit unitary. *Acts on 1 qubit.*

### (d) Hadamard Gate
Creates an equal superposition of |0⟩ and |1⟩. Equivalent to a balanced beam splitter in the polarization basis and the starting point for almost every entangling circuit. *Acts on 1 qubit.*

### (e) 50:50 Beam Splitter (BS)
Coherently mixes two spatial modes with equal probability. Described by a unitary transformation across two modes — the key ingredient of Hong-Ou-Mandel interference and the foundation of linear optical quantum computing. *Acts on 2 qubits.*

### (f) Polarizing Beam Splitter (PBS)
Separates photons by polarization — |H⟩ (horizontal) transmits, |V⟩ (vertical) reflects. Used for polarization-based qubit routing, Bell state measurement, and entanglement distribution. *Acts on 2 qubits.*

### (g) CNOT Gate
Controlled-NOT: flips the target qubit if and only if the control qubit is |1⟩. The fundamental entangling gate. Together with single-qubit rotations, CNOTs form a universal gate set. *Acts on 2 qubits (control, target).*

### (h) Threshold Detector
Clicks when one or more photons are present, regardless of exact number. Used for heralding — a click in an ancilla mode signals that the desired state was prepared in the signal mode, enabling probabilistic post-selection. *Acts on 1 qubit.*

### (i) Photon-Number-Resolving (PNR) Detector
Measures the exact number of photons in a mode. More powerful than threshold detectors for post-selection; allows rejection of multi-photon events that corrupt qubit encodings. *Acts on 1 qubit.*

### (j) SPDC Photon-Pair Source
The entanglement engine. Spontaneous Parametric Down-Conversion splits a pump photon into two correlated photons in the Bell state |Φ+⟩ = (1/√2)(|00⟩ + |11⟩). The standard lab method for generating entangled photon pairs. **Limited to 3 sources per circuit.** *Acts on 2 qubits.*

### (k) Cross-Kerr Nonlinear Crystal
Applies a conditional phase e^{iχ} only when both input modes are occupied — a photon-photon interaction. Theoretically enables deterministic two-qubit gates without measurement tricks, though experimentally very challenging. *Acts on 2 qubits.*

### Heralding and Post-Selection (Strategy, not a component)
By placing detectors on ancilla modes and conditioning on their outcomes, the circuit can probabilistically prepare high-quality target states. The environment models this via projective measurement: the chosen qubit is projected onto |0⟩ (no-click herald) and the remaining state is renormalised.

---

## Resource Constraints

The agent must respect these limits — they reflect both the problem statement and real experimental practicalities.

| Resource | Limit | Reason |
|---|---|---|
| SPDC sources | max 3 | Expensive, noisy, hard to align in real labs |
| Components per circuit | max 20 | Prevents degenerate infinite-length solutions |
| Spatial modes | max 8 | Table space and mode-matching constraints |
| Qubits | 1 to 4 | Scope of the challenge |

---

## Reward Design

Getting the reward function right is the trickiest part of any RL project. Too sparse and the agent never learns; too dense and it learns to game the system.

**Per-step reward** (applied after every component placement):
```
R_step = (fidelity_now − fidelity_prev) + 0.1 × fidelity_now² − 0.01
```
The small step penalty (−0.01) pushes toward shorter, more resource-efficient circuits. The squared fidelity term gives disproportionately higher reward as the agent approaches the target, sharpening the reward landscape near solutions.

**Terminal reward** (applied at end of each episode):
```
If fidelity > 0.99:    R_terminal = 2.0 + improvement          ← jackpot
If improving at all:   R_terminal = improvement² × 4 + improvement
If no improvement:     R_terminal = improvement − 0.1          ← penalty
```

`improvement` is always relative to the *baseline* fidelity at episode start, so the agent cannot exploit a lucky initialisation without earning it.

---

## Training Details

```bash
# Standard run — 4-qubit GHZ
python train_ppo.py --target ghz --num_qubits 4 --timesteps 200000

# Faster run — 3-qubit GHZ
python train_ppo.py --target ghz --num_qubits 3 --timesteps 200000

# Smallest / fastest — Bell state
python train_ppo.py --target bell --num_qubits 2 --timesteps 100000

# Custom state from file
python train_ppo.py --target my_state.npy --num_qubits 4 --timesteps 300000

# Override max steps per episode
python train_ppo.py --target ghz --num_qubits 4 --max_steps 10 --timesteps 200000
```

Default `max_steps` scales with qubit count so simpler problems have tighter budgets:

| Qubits | Default max steps | Approx action space size |
|---|---|---|
| 1 | 5 | ~3 |
| 2 | 8 | ~20 |
| 3 | 12 | ~60 |
| 4 | 15 | ~120 |

### PPO Hyperparameters

| Parameter | Value | What it controls |
|---|---|---|
| `learning_rate` | 3e-4 | Adam optimizer step size |
| `n_steps` | 4096 | Rollout buffer length before each update |
| `batch_size` | 64 | Mini-batch size for gradient steps |
| `n_epochs` | 10 | Gradient passes per collected rollout |
| `gamma` | 0.99 | Discount factor — values future fidelity gains |
| `gae_lambda` | 0.95 | Bias-variance tradeoff in advantage estimation |
| `ent_coef` | 0.01 | Entropy bonus — keeps exploration alive |
| `clip_range` | 0.2 | PPO clipping — stabilises policy updates |
| `max_grad_norm` | 0.3 | Gradient clipping — prevents exploding updates |

### Early Stopping

Training automatically stops early when the average fidelity over the last 100 episodes stays above **0.99** for **5 consecutive** logging intervals. This prevents catastrophic forgetting after convergence. You'll see it logged as:

```
[EarlyStop] Avg fidelity 0.9923 >= 0.99 (3/5)
...
[EarlyStop] CONVERGED — stopping at step 87500
```

### Checkpointing

Models are saved every `timesteps / 10` steps (minimum every 1000 steps) to `./models/`. The best model by evaluation return is separately tracked as `best_model.zip`.

---

## Evaluation

```bash
# Evaluate the best saved model
python train_ppo.py --evaluate --model_path ./models/best_model.zip --num_qubits 4

# More episodes for statistical confidence
python train_ppo.py --evaluate --model_path ./models/best_model.zip --num_qubits 3 --eval_episodes 20

# Via main.py
python main.py --mode eval --model ./models/best_model.zip --target ghz --num_qubits 4
```

The evaluation loop prints fidelity and component count per episode, then displays the best circuit found — showing exactly which components were placed in which order.

---

## Reproducing Results

```bash
# 1. Clone
git clone <your-repo-url>
cd quantum_rl_fixed_copy

# 2. Install (Python 3.10+ recommended)
pip install -r requirements.txt

# 3. Verify physics engine
python quantum_physics.py

# 4. Verify component library
python quantum_components.py

# 5. Run environment smoke test
python quantum_env.py

# 6. Verify known-optimal GHZ circuit gives fidelity = 1.0
python main.py --mode known --target ghz --num_qubits 4

# 7. Full training run
python train_ppo.py --target ghz --num_qubits 4 --timesteps 200000 --seed 42

# 8. Evaluate
python train_ppo.py --evaluate --model_path ./models/best_model.zip --num_qubits 4
```

All training runs use `seed=42` by default.

---

## All CLI Arguments

### `train_ppo.py`

| Argument | Default | Description |
|---|---|---|
| `--target` | `ghz` | Target state: preset name, file path, or inline vector |
| `--num_qubits` | `4` | Number of qubits (1–4). Also reads `NUM_QUBITS` env var |
| `--max_steps` | auto | Max components per episode. Auto: 1→5, 2→8, 3→12, 4→15 |
| `--timesteps` | `200000` | Total training timesteps |
| `--lr` | `3e-4` | Learning rate |
| `--batch_size` | `64` | Mini-batch size |
| `--seed` | `42` | Random seed |
| `--save_dir` | `./models` | Where to save checkpoints |
| `--log_dir` | `./logs` | Where to save training logs |
| `--log_freq` | `5000` | Print stats every N steps |
| `--evaluate` | flag | Evaluate a saved model instead of training |
| `--model_path` | `./models/best_model.zip` | Model to load for evaluation |
| `--eval_episodes` | `10` | Number of evaluation episodes |

### `main.py`

| Argument | Default | Description |
|---|---|---|
| `--mode` | `demo` | `train` / `demo` / `eval` / `known` |
| `--target` | `ghz` | Target state |
| `--num_qubits` | `4` | Number of qubits |
| `--max_steps` | auto | Max steps per episode |
| `--timesteps` | `100000` | Timesteps for `train` mode |
| `--model` | `./models/best_model.zip` | Model path for `eval` mode |
| `--episodes` | `5` | Episodes for `demo` / `eval` modes |

---

## Dependencies

```
numpy>=1.24.0
gymnasium>=0.29.0
stable-baselines3>=2.1.0
torch>=2.0.0
tqdm>=4.65.0
```

Python 3.10 or later. A GPU is not required — the Hilbert space stays small (max 16-dimensional for 4 qubits), so CPU training is perfectly fast.

---

## Design Choices and Limitations

A few honest notes on where this implementation sits:

- **Ideal conditions only.** The simulator assumes no photon loss, perfect detectors, and perfect gate fidelity. Real experiments are far messier — loss and noise would significantly reduce achievable fidelity.
- **Discrete action space.** Component parameters (like HWP angle θ) are discretised rather than continuously optimised. A hybrid approach — RL to find circuit structure, then gradient optimisation to tune parameters — could push fidelity higher.
- **State vector simulation.** We work with pure state vectors, which scales exponentially with qubit count. Fine up to 4 qubits (16-dimensional) but would need density matrix or tensor network methods beyond that.
- **Simplified post-selection.** Heralding is modelled as projective measurement onto |0⟩ with renormalisation, rather than a full coincidence-counting simulation over many shots.

---

## Acknowledgements

Built for the **Physis Techne Symposium 2026** ML challenge on automated quantum optical experiment design.

---

