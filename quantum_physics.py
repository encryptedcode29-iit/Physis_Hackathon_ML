import numpy as np
import os

NUM_QUBITS: int = int(os.getenv("NUM_QUBITS", 4))
DIM: int = 2 ** NUM_QUBITS
def set_num_qubits(n: int) -> None:
    global NUM_QUBITS, DIM
    if n < 1 or n > 4:
        raise ValueError(f"num_qubits must be 1-4, got {n}")
    NUM_QUBITS = n
    DIM = 2 ** n


def get_num_qubits() -> int:
    return NUM_QUBITS
KET_0 = np.array([1.0, 0.0], dtype=np.complex128)
KET_1 = np.array([0.0, 1.0], dtype=np.complex128)

I2 = np.eye(2, dtype=np.complex128)


def computational_basis_state(index: int) -> np.ndarray:
    state = np.zeros(DIM, dtype=np.complex128)
    state[index] = 1.0
    return state


def zero_state() -> np.ndarray:
    return computational_basis_state(0)


def hadamard_2x2() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def phase_shift_2x2(phi: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)


def half_wave_plate_2x2(theta: float) -> np.ndarray:
    c = np.cos(2 * theta)
    s = np.sin(2 * theta)
    return np.array([[c, s], [s, -c]], dtype=np.complex128)


def quarter_wave_plate_2x2(theta: float) -> np.ndarray:
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    return (1 / np.sqrt(2)) * np.array(
        [[1 + 1j * c2, 1j * s2], [1j * s2, 1 - 1j * c2]], dtype=np.complex128
    )


def pauli_x_2x2() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def pauli_z_2x2() -> np.ndarray:
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def cnot_4x4() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dtype=np.complex128,
    )


def beam_splitter_4x4() -> np.ndarray:
    bs = np.eye(4, dtype=np.complex128)
    bs[0, 0] = 1 / np.sqrt(2)
    bs[0, 1] = 1j / np.sqrt(2)
    bs[1, 0] = 1j / np.sqrt(2)
    bs[1, 1] = 1 / np.sqrt(2)
    bs[2, 2] = 1 / np.sqrt(2)
    bs[2, 3] = 1j / np.sqrt(2)
    bs[3, 2] = 1j / np.sqrt(2)
    bs[3, 3] = 1 / np.sqrt(2)
    return bs


def polarizing_beam_splitter_4x4() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]],
        dtype=np.complex128,
    )


def cross_kerr_4x4(chi: float) -> np.ndarray:
    ck = np.eye(4, dtype=np.complex128)
    ck[3, 3] = np.exp(1j * chi)
    return ck


def _single_qubit_gate_16(gate_2x2: np.ndarray, target: int) -> np.ndarray:
    assert 0 <= target < NUM_QUBITS, f"target qubit must be 0-3, got {target}"
    ops = [I2] * NUM_QUBITS
    ops[target] = gate_2x2
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _two_qubit_gate_16(gate_4x4: np.ndarray, q1: int, q2: int) -> np.ndarray:
    assert 0 <= q1 < NUM_QUBITS and 0 <= q2 < NUM_QUBITS, "qubits must be 0-3"
    assert q1 != q2, "qubits must be different"

    full = np.zeros((DIM, DIM), dtype=np.complex128)

    for col in range(DIM):
        bits = [(col >> (NUM_QUBITS - 1 - i)) & 1 for i in range(NUM_QUBITS)]
        b1, b2 = bits[q1], bits[q2]

        in_idx = b1 * 2 + b2

        for out_2q in range(4):
            coeff = gate_4x4[out_2q, in_idx]
            if abs(coeff) < 1e-15:
                continue
            new_bits = bits.copy()
            new_bits[q1] = (out_2q >> 1) & 1
            new_bits[q2] = out_2q & 1
            row = 0
            for i, b in enumerate(new_bits):
                row |= b << (NUM_QUBITS - 1 - i)
            full[row, col] += coeff

    return full


def Hadamard(target: int) -> np.ndarray:
    return _single_qubit_gate_16(hadamard_2x2(), target)


def PhaseShift(target: int, phi: float) -> np.ndarray:
    return _single_qubit_gate_16(phase_shift_2x2(phi), target)


def HWP(target: int, theta: float) -> np.ndarray:
    return _single_qubit_gate_16(half_wave_plate_2x2(theta), target)


def QWP(target: int, theta: float) -> np.ndarray:
    return _single_qubit_gate_16(quarter_wave_plate_2x2(theta), target)


def PauliX(target: int) -> np.ndarray:
    return _single_qubit_gate_16(pauli_x_2x2(), target)


def PauliZ(target: int) -> np.ndarray:
    return _single_qubit_gate_16(pauli_z_2x2(), target)


def CNOT(control: int, target: int) -> np.ndarray:
    return _two_qubit_gate_16(cnot_4x4(), control, target)


def BeamSplitter(q1: int, q2: int) -> np.ndarray:
    return _two_qubit_gate_16(beam_splitter_4x4(), q1, q2)


def PBS(q1: int, q2: int) -> np.ndarray:
    return _two_qubit_gate_16(polarizing_beam_splitter_4x4(), q1, q2)


def CrossKerr(q1: int, q2: int, chi: float) -> np.ndarray:
    return _two_qubit_gate_16(cross_kerr_4x4(chi), q1, q2)


def apply_gate(state: np.ndarray, gate: np.ndarray) -> np.ndarray:
    new_state = gate @ state
    norm = np.linalg.norm(new_state)
    if norm > 1e-12:
        new_state /= norm
    return new_state


def fidelity(state: np.ndarray, target: np.ndarray) -> float:
    overlap = np.vdot(target, state) 
    return float(np.abs(overlap) ** 2)


def ghz_state() -> np.ndarray:
    state = np.zeros(DIM, dtype=np.complex128)
    state[0] = 1 / np.sqrt(2)  
    state[DIM - 1] = 1 / np.sqrt(2) 
    return state


def w_state() -> np.ndarray:
    state = np.zeros(DIM, dtype=np.complex128)
    amp = 1.0 / np.sqrt(NUM_QUBITS)
    for i in range(NUM_QUBITS):
        state[1 << i] = amp
    return state


def bell_phi_plus(q1: int = 0, q2: int = 1) -> np.ndarray:
    state = np.zeros(DIM, dtype=np.complex128)
    state[0] = 1 / np.sqrt(2)
    bits_11 = (1 << (NUM_QUBITS - 1 - q1)) | (1 << (NUM_QUBITS - 1 - q2))
    state[bits_11] = 1 / np.sqrt(2)
    return state


def load_target_from_file(filepath: str) -> np.ndarray:
    filepath = filepath.strip()
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Target state file not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == ".npy":
        state = np.load(filepath).astype(np.complex128).flatten()
    elif ext in (".txt", ".csv", ".dat"):
        with open(filepath, "r") as f:
            content = f.read().strip()
        content = content.replace(",", " ").replace(";", " ")
        parts = content.split()

        if len(parts) == DIM * 2:
            state = np.array(
                [float(parts[2 * k]) + 1j * float(parts[2 * k + 1])
                 for k in range(DIM)],
                dtype=np.complex128,
            )
        else:
            state = np.array([complex(p) for p in parts], dtype=np.complex128)
    else:
        state = np.loadtxt(filepath, dtype=np.complex128).flatten()
    
    if len(state) != DIM:
        raise ValueError(
            f"Target state must have {DIM} elements (got {len(state)}). "
            f"This engine supports up to {NUM_QUBITS} qubits."
        )
    
    norm = np.linalg.norm(state)
    if norm < 1e-12:
        raise ValueError("Target state has zero norm (all zeros).")
    state /= norm
    
    return state


def parse_target_vector(vector_str: str) -> np.ndarray:
    vector_str = vector_str.strip()
    parts = vector_str.replace(",", " ").split()
    state = np.array([complex(p) for p in parts], dtype=np.complex128)
    
    if len(state) != DIM:
        raise ValueError(
            f"Target state must have {DIM} elements (got {len(state)}). "
            f"This engine supports up to {NUM_QUBITS} qubits."
        )
    
    norm = np.linalg.norm(state)
    if norm < 1e-12:
        raise ValueError("Target state has zero norm.")
    state /= norm
    
    return state


def build_target_state(name: str) -> np.ndarray:
    name = name.strip()
    name_lower = name.lower()
    
    if name_lower in ("ghz", "ghz4"):
        return ghz_state()
    elif name_lower in ("w", "w4"):
        return w_state()
    elif name_lower in ("bell", "bell_phi_plus", "phi+"):
        return bell_phi_plus()
    
    if os.path.isfile(name):
        return load_target_from_file(name)
    
    try:
        return parse_target_vector(name)
    except (ValueError, TypeError):
        pass
    
    raise ValueError(
        f"Cannot interpret target '{name}'. Expected one of:\n"
        f"  - Named preset: ghz, w, bell\n"
        f"  - File path: /path/to/state.npy or state.txt\n"
        f"  - Inline vector: '0.707 0 0 ... 0 0.707' (16 complex numbers)"
    )


def spdc_prepare(state: np.ndarray, q1: int, q2: int) -> np.ndarray:
    h_gate = Hadamard(q1)
    cnot_gate = CNOT(q1, q2)

    result = apply_gate(state, h_gate)
    result = apply_gate(result, cnot_gate)

    norm = np.linalg.norm(result)
    if norm > 1e-12:
        result /= norm

    return result

def _verify_unitary(name: str, gate: np.ndarray, tol: float = 1e-10) -> bool:
    product = gate @ gate.conj().T
    identity = np.eye(gate.shape[0], dtype=np.complex128)
    ok = np.allclose(product, identity, atol=tol)
    if not ok:
        max_err = np.max(np.abs(product - identity))
        print(f"[WARN] {name} is NOT unitary (max error: {max_err:.3e})")
    return ok


if __name__ == "__main__":
    print("=== Quantum Physics Engine Self-Test ===\n")

    for q in range(NUM_QUBITS):
        _verify_unitary(f"Hadamard(q={q})", Hadamard(q))
        _verify_unitary(f"PhaseShift(q={q}, π/4)", PhaseShift(q, np.pi / 4))
        _verify_unitary(f"HWP(q={q}, π/8)", HWP(q, np.pi / 8))
        _verify_unitary(f"QWP(q={q}, π/8)", QWP(q, np.pi / 8))
        _verify_unitary(f"PauliX(q={q})", PauliX(q))
        _verify_unitary(f"PauliZ(q={q})", PauliZ(q))

    for q1, q2 in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        _verify_unitary(f"CNOT({q1},{q2})", CNOT(q1, q2))
        _verify_unitary(f"BeamSplitter({q1},{q2})", BeamSplitter(q1, q2))
        _verify_unitary(f"PBS({q1},{q2})", PBS(q1, q2))
        _verify_unitary(f"CrossKerr({q1},{q2}, π/4)", CrossKerr(q1, q2, np.pi / 4))

    psi = zero_state()
    psi = apply_gate(psi, Hadamard(0))
    psi = apply_gate(psi, CNOT(0, 1))
    psi = apply_gate(psi, CNOT(1, 2))
    psi = apply_gate(psi, CNOT(2, 3))
    ghz = ghz_state()
    f = fidelity(psi, ghz)
    print(f"\nGHZ fidelity (manual circuit): {f:.6f}  {'PASS' if f > 0.999 else 'FAIL'}")
    print(f"GHZ target state:  {ghz}")
    print(f"Prepared state:    {psi}")
    print("\n[OK] All self-tests passed!" if f > 0.999 else "\n[FAIL] Some tests failed.")
