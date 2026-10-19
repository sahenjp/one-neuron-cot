"""arXiv:2610.17283 の単一ニューロン自己回帰CoTシミュレーション検証パッケージ。"""
from .sidon import sidon_golomb
from .circuits import (
    Circuit,
    Gate,
    adder_circuit,
    mlp_relu_circuit,
    parity_circuit,
    random_threshold_circuit,
)
from .simulate import (
    ArPlan,
    build_neuron,
    designated_values,
    run_fast,
    run_reference,
)
from .learn import fit_cot_perceptron

__all__ = [
    "sidon_golomb",
    "Circuit",
    "Gate",
    "parity_circuit",
    "adder_circuit",
    "mlp_relu_circuit",
    "random_threshold_circuit",
    "ArPlan",
    "build_neuron",
    "run_fast",
    "run_reference",
    "designated_values",
    "fit_cot_perceptron",
]
