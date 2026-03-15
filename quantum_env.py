import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from typing import Optional, Dict, Any, Tuple

import quantum_physics as qp
import quantum_components as qc

_MAX_STEPS_DEFAULT = {1: 5, 2: 8, 3: 12, 4: 15}


class QuantumOpticsEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        target_state_name: str = "ghz",
        num_qubits: int = int(os.getenv("NUM_QUBITS", 4)),
        max_steps: Optional[int] = None,
        reward_shaping: bool = True,
        step_penalty: float = 0.01,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if num_qubits < 1 or num_qubits > 4:
            raise ValueError(f"num_qubits must be 1-4, got {num_qubits}")

        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits

        if max_steps is None:
            max_steps = _MAX_STEPS_DEFAULT[num_qubits]

        qp.set_num_qubits(num_qubits)
        qc.rebuild_catalogue()

        self.target_state_name = target_state_name
        self.target_state = qp.build_target_state(target_state_name)
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.step_penalty = step_penalty
        self.render_mode = render_mode

        self.action_catalogue = qc.ACTION_CATALOGUE
        self.num_actions = qc.NUM_ACTIONS

        self.action_space = spaces.Discrete(self.num_actions)

        self.EMPTY_SLOT = self.num_actions 

        self.observation_space = spaces.Dict({
            "circuit_sequence": spaces.MultiDiscrete(
                [self.num_actions + 1] * self.max_steps  
            ),
            "step_count": spaces.Discrete(self.max_steps + 1),
            "quantum_state_real": spaces.Box(
                low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32
            ),
            "quantum_state_imag": spaces.Box(
                low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32
            ),
            "target_state_real": spaces.Box(
                low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32
            ),
            "target_state_imag": spaces.Box(
                low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32
            ),
            "current_fidelity": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
        })

        self.quantum_state: np.ndarray = qp.zero_state()
        self.circuit: list = []
        self.circuit_ids: np.ndarray = np.full(self.max_steps, self.EMPTY_SLOT, dtype=np.int64)
        self.current_step: int = 0
        self.resource_tracker = qc.ResourceTracker()
        self.prev_fidelity: float = 0.0
        self.baseline_fidelity: float = 0.0  

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        qp.set_num_qubits(self.num_qubits)  

        self.quantum_state = qp.zero_state()
        self.circuit = []
        self.circuit_ids = np.full(self.max_steps, self.EMPTY_SLOT, dtype=np.int64)
        self.current_step = 0
        self.resource_tracker.reset()
        self.baseline_fidelity = qp.fidelity(self.quantum_state, self.target_state)
        self.prev_fidelity = self.baseline_fidelity

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        terminated = False
        truncated = False
        reward = 0.0
        qp.set_num_qubits(self.num_qubits)  

        if action < 0 or action >= self.num_actions:
            reward = -0.1
            self.current_step += 1
            if self.current_step >= self.max_steps:
                truncated = True
                fid = qp.fidelity(self.quantum_state, self.target_state)
                reward += self._terminal_reward(fid)
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        component = self.action_catalogue[action]

        can_add, reason = self.resource_tracker.can_add(component)
        if not can_add:
            reward = -0.05
            self.current_step += 1
            if self.current_step >= self.max_steps:
                truncated = True
                fid = qp.fidelity(self.quantum_state, self.target_state)
                reward += self._terminal_reward(fid)
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        self._apply_component(component, action)

        current_fidelity = qp.fidelity(self.quantum_state, self.target_state)

        if self.reward_shaping:
            fidelity_delta = current_fidelity - self.prev_fidelity
            reward = fidelity_delta + (current_fidelity ** 2) * 0.1 - self.step_penalty
        else:
            reward = -self.step_penalty

        self.prev_fidelity = current_fidelity

        if current_fidelity > 0.9999:
            terminated = True
            reward += self._terminal_reward(current_fidelity)
        elif self.current_step >= self.max_steps:
            truncated = True
            reward += self._terminal_reward(current_fidelity)

        return self._get_obs(), reward, terminated, truncated, self._get_info(fidelity=current_fidelity)

    def render(self) -> Optional[str]:
        if self.render_mode == "human":
            print(self._render_text())
        elif self.render_mode == "ansi":
            return self._render_text()
        return None

    def _apply_component(self, component: qc.QuantumComponent, action_id: int):
        if component.comp_type == qc.ComponentType.SPDC_SOURCE:
            q1, q2 = component.qubits
            self.quantum_state = qp.spdc_prepare(self.quantum_state, q1, q2)

        elif component.comp_type in (
            qc.ComponentType.THRESHOLD_DETECTOR,
            qc.ComponentType.PNR_DETECTOR,
        ):
            q = component.qubits[0]
            self.quantum_state = self._project_qubit(self.quantum_state, q, outcome=0)

        elif component.gate_matrix is not None:
            self.quantum_state = qp.apply_gate(self.quantum_state, component.gate_matrix)

        self.circuit.append(component)
        self.circuit_ids[self.current_step] = action_id
        self.current_step += 1
        self.resource_tracker.add(component)

    def _project_qubit(self, state: np.ndarray, qubit: int, outcome: int) -> np.ndarray:
        projected = np.zeros_like(state)
        for idx in range(self.dim):
            bits = [(idx >> (self.num_qubits - 1 - i)) & 1 for i in range(self.num_qubits)]
            if bits[qubit] == outcome:
                projected[idx] = state[idx]

        norm = np.linalg.norm(projected)
        if norm > 1e-12:
            projected /= norm
        else:
            projected = state.copy()

        return projected

    def _terminal_reward(self, fidelity_val: float) -> float:
        improvement = fidelity_val - self.baseline_fidelity
        
        if fidelity_val > 0.99:
            return 2.0 + improvement
        elif improvement > 0:
            return improvement ** 2 * 4.0 + improvement
        else:
            return improvement - 0.1

    def _get_obs(self) -> Dict[str, np.ndarray]:
        fid = qp.fidelity(self.quantum_state, self.target_state)
        return {
            "circuit_sequence": self.circuit_ids.copy(),
            "step_count": np.int64(self.current_step),
            "quantum_state_real": self.quantum_state.real.astype(np.float32),
            "quantum_state_imag": self.quantum_state.imag.astype(np.float32),
            "target_state_real": self.target_state.real.astype(np.float32),
            "target_state_imag": self.target_state.imag.astype(np.float32),
            "current_fidelity": np.array([fid], dtype=np.float32),
        }

    def _get_info(self, fidelity: Optional[float] = None) -> Dict[str, Any]:
        if fidelity is None:
            fidelity = qp.fidelity(self.quantum_state, self.target_state)
        return {
            "fidelity": fidelity,
            "num_components": self.current_step,
            "circuit": [str(c) for c in self.circuit],
            "spdc_count": self.resource_tracker.spdc_count,
            "num_qubits": self.num_qubits,
        }

    def _render_text(self) -> str:
        lines = [
            f"=== Quantum Optical Circuit ({self.num_qubits}q, step {self.current_step}/{self.max_steps}) ===",
            f"Target: {self.target_state_name.upper()}",
            f"Fidelity: {qp.fidelity(self.quantum_state, self.target_state):.6f}",
            "",
            "Components placed:",
        ]
        for i, comp in enumerate(self.circuit):
            lines.append(f"  {i + 1}. {comp}")
        if not self.circuit:
            lines.append("  (empty)")
        lines.append("")
        return "\n".join(lines)


def register_env():
    gym.register(
        id="QuantumOptics-v0",
        entry_point="quantum_env:QuantumOpticsEnv",
        max_episode_steps=15,
    )


if __name__ == "__main__":
    print("=== Quantum Optics Environment Smoke Test ===\n")

    env = QuantumOpticsEnv(target_state_name="ghz", max_steps=15, render_mode="human")
    obs, info = env.reset()
    print(f"Action space:      Discrete({env.action_space.n})")
    print(f"Observation keys:  {list(obs.keys())}")
    print(f"Circuit seq shape: {obs['circuit_sequence'].shape}")
    print(f"Initial fidelity:  {info['fidelity']:.6f}\n")

    total_reward = 0
    for step in range(15):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.render()
    print(f"Total reward:  {total_reward:.4f}")
    print(f"Final fidelity: {info['fidelity']:.6f}")
    print(f"Components used: {info['num_components']}")
    print("\n[OK] Smoke test passed!")