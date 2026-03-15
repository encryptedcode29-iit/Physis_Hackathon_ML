import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum, auto

import quantum_physics as qp


class ComponentType(Enum):
    PHASE_SHIFTER = auto()
    HALF_WAVE_PLATE = auto()
    QUARTER_WAVE_PLATE = auto()
    BEAM_SPLITTER = auto()
    POLARIZING_BEAM_SPLITTER = auto()
    CNOT_GATE = auto()
    HADAMARD = auto()
    THRESHOLD_DETECTOR = auto()
    PNR_DETECTOR = auto()
    SPDC_SOURCE = auto()
    CROSS_KERR = auto()


@dataclass
class QuantumComponent:
    comp_type: ComponentType
    name: str
    description: str
    qubits: Tuple[int, ...]
    params: Dict[str, float] = field(default_factory=dict)
    gate_matrix: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self):
        q_str = ",".join(str(q) for q in self.qubits)
        p_str = ", ".join(f"{k}={v:.3f}" for k, v in self.params.items())
        extra = f" ({p_str})" if p_str else ""
        return f"{self.name}[q{q_str}]{extra}"


def phase_shifter(target: int, phi: float) -> QuantumComponent:
    return QuantumComponent(
        comp_type=ComponentType.PHASE_SHIFTER,
        name="PhaseShifter",
        description="Applies an arbitrary phase shift e^{iφ} to a single optical mode.",
        qubits=(target,),
        params={"phi": phi},
        gate_matrix=qp.PhaseShift(target, phi),
    )


def half_wave_plate(target: int, theta: float) -> QuantumComponent:
    return QuantumComponent(
        comp_type=ComponentType.HALF_WAVE_PLATE,
        name="HWP",
        description=(
            "Half-wave plate for polarization manipulation. "
            "Rotates the polarization state by 2θ around the optical axis."
        ),
        qubits=(target,),
        params={"theta": theta},
        gate_matrix=qp.HWP(target, theta),
    )


def quarter_wave_plate(target: int, theta: float) -> QuantumComponent:
    return QuantumComponent(
        comp_type=ComponentType.QUARTER_WAVE_PLATE,
        name="QWP",
        description=(
            "Quarter-wave plate for converting between linear and circular "
            "polarization. Introduces a π/2 relative phase between orthogonal components."
        ),
        qubits=(target,),
        params={"theta": theta},
        gate_matrix=qp.QWP(target, theta),
    )


def hadamard_gate(target: int) -> QuantumComponent:
    return QuantumComponent(
        comp_type=ComponentType.HADAMARD,
        name="Hadamard",
        description=(
            "Creates an equal superposition of |0⟩ and |1⟩. "
            "Equivalent to a balanced beam splitter in the polarization basis."
        ),
        qubits=(target,),
        gate_matrix=qp.Hadamard(target),
    )


def beam_splitter(q1: int, q2: int) -> QuantumComponent:
    return QuantumComponent(
        comp_type=ComponentType.BEAM_SPLITTER,
        name="BS",
        description=(
            "50:50 beam splitter for coherent mixing of two spatial modes. "
            "Splits an incoming photon into a superposition across two output ports."
        ),
        qubits=(q1, q2),
        gate_matrix=qp.BeamSplitter(q1, q2),
    )


def polarizing_beam_splitter(q1: int, q2: int) -> QuantumComponent:
    return QuantumComponent(
        comp_type=ComponentType.POLARIZING_BEAM_SPLITTER,
        name="PBS",
        description=(
            "Polarizing beam splitter that transmits horizontally polarized "
            "photons (|H⟩/|0⟩) and reflects vertically polarized photons (|V⟩/|1⟩), "
            "separating photons based on their polarization state."
        ),
        qubits=(q1, q2),
        gate_matrix=qp.PBS(q1, q2),
    )


def cnot_gate(control: int, target: int) -> QuantumComponent:
    return QuantumComponent(
        comp_type=ComponentType.CNOT_GATE,
        name="CNOT",
        description=(
            "Controlled-NOT entangling gate. Flips the target qubit if the "
            "control qubit is |1⟩. Fundamental for creating entanglement."
        ),
        qubits=(control, target),
        gate_matrix=qp.CNOT(control, target),
    )


def cross_kerr_crystal(q1: int, q2: int, chi: float) -> QuantumComponent:
    return QuantumComponent(
        comp_type=ComponentType.CROSS_KERR,
        name="CrossKerr",
        description=(
            "Cross-Kerr nonlinear crystal that induces a conditional phase shift. "
            "Applies phase e^{iχ} only when both input modes are occupied, "
            "enabling photon-photon interaction."
        ),
        qubits=(q1, q2),
        params={"chi": chi},
        gate_matrix=qp.CrossKerr(q1, q2, chi),
    )


def spdc_source(q1: int, q2: int) -> QuantumComponent:
    return QuantumComponent(
        comp_type=ComponentType.SPDC_SOURCE,
        name="SPDC",
        description=(
            "Spontaneous Parametric Down-Conversion (SPDC) photon-pair source "
            "using a phase-matched nonlinear crystal. Produces entangled photon "
            "pairs in the Bell state |Φ+⟩ = (1/√2)(|00⟩+|11⟩). Assumes ideal "
            "single-pair emission."
        ),
        qubits=(q1, q2),
        gate_matrix=None,  
    )


def threshold_detector(target: int) -> QuantumComponent:
    return QuantumComponent(
        comp_type=ComponentType.THRESHOLD_DETECTOR,
        name="ThresholdDet",
        description=(
            "Threshold detector that registers a 'click' when one or more "
            "photons are present. Used for heralding, post-selection, and "
            "probabilistic state preparation schemes."
        ),
        qubits=(target,),
        gate_matrix=None,
    )


def pnr_detector(target: int) -> QuantumComponent:
    return QuantumComponent(
        comp_type=ComponentType.PNR_DETECTOR,
        name="PNR_Det",
        description=(
            "Photon-number-resolving (PNR) detector that measures the exact "
            "number of photons in a mode. Enables more sophisticated "
            "post-selection and state characterization."
        ),
        qubits=(target,),
        gate_matrix=None,  
    )


MAX_SPDC_SOURCES = 3
MAX_COMPONENTS = 20
MAX_SPATIAL_MODES = 8


class ResourceTracker:

    def __init__(self):
        self.spdc_count: int = 0
        self.total_components: int = 0
        self.spatial_modes_used: set = set()

    def can_add(self, component: QuantumComponent) -> Tuple[bool, str]:
        if self.total_components >= MAX_COMPONENTS:
            return False, f"Max components ({MAX_COMPONENTS}) reached."

        if component.comp_type == ComponentType.SPDC_SOURCE:
            if self.spdc_count >= MAX_SPDC_SOURCES:
                return False, f"Max SPDC sources ({MAX_SPDC_SOURCES}) reached."

        new_modes = self.spatial_modes_used | set(component.qubits)
        if len(new_modes) > MAX_SPATIAL_MODES:
            return False, f"Max spatial modes ({MAX_SPATIAL_MODES}) would be exceeded."

        return True, "OK"

    def add(self, component: QuantumComponent):
        self.total_components += 1
        self.spatial_modes_used.update(component.qubits)
        if component.comp_type == ComponentType.SPDC_SOURCE:
            self.spdc_count += 1

    def reset(self):
        self.spdc_count = 0
        self.total_components = 0
        self.spatial_modes_used = set()


def build_action_catalogue() -> List[QuantumComponent]:
    n = qp.NUM_QUBITS
    actions: List[QuantumComponent] = []

    for q in range(n):
        actions.append(hadamard_gate(q))
        actions.append(phase_shifter(q, np.pi / 2))
        actions.append(half_wave_plate(q, np.pi / 4))

    if n >= 2:
        qubit_pairs    = [(i, j) for i in range(n) for j in range(n) if i != j]
        adjacent_pairs = [(i, i + 1) for i in range(n - 1)] + [(0, n - 1)]
        spdc_pairs     = [(i, j) for i in range(n) for j in range(i + 1, n)]

        for q1, q2 in qubit_pairs:
            actions.append(cnot_gate(q1, q2))
        for q1, q2 in adjacent_pairs:
            actions.append(beam_splitter(q1, q2))
        for q1, q2 in spdc_pairs:
            actions.append(spdc_source(q1, q2))

    return actions


ACTION_CATALOGUE = build_action_catalogue()
NUM_ACTIONS = len(ACTION_CATALOGUE)


def rebuild_catalogue() -> None:
    global ACTION_CATALOGUE, NUM_ACTIONS
    ACTION_CATALOGUE = build_action_catalogue()
    NUM_ACTIONS = len(ACTION_CATALOGUE)


if __name__ == "__main__":
    print(f"=== Quantum Component Library ===")
    print(f"Total discrete actions: {NUM_ACTIONS}\n")

    from collections import Counter
    type_counts = Counter(a.comp_type.name for a in ACTION_CATALOGUE)
    for t, c in sorted(type_counts.items()):
        print(f"  {t:30s}: {c} actions")

    print("\nFirst 10 actions:")
    for i, a in enumerate(ACTION_CATALOGUE[:10]):
        print(f"  [{i:3d}] {a}")