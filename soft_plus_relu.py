"""
Softplus via quadrature of the quantum sigmoid circuit + exact ReLU via CSWAP.

Corrected softplus construction
--------------------------------
The product-state heuristic of the earlier draft is replaced by an exact
identity: since d/dx softplus(x) = sigma(x),
    softplus(x) = \\int_{-\\infty}^{x} sigma(t) dt .
We evaluate sigma(t) with the exact single-qubit geometric circuit (sigmoid_tanh.py):
sigma(x) = (1 + <Z_0>)/2 exactly, so the only per-node error is shot noise.
at quadrature nodes on [-T, x] and integrate numerically (trapezoid).
For x > 0 we use softplus(x) = x + softplus(-x).

ReLU
----
For basis-encoded signed integers, one CSWAP per magnitude qubit, controlled
by the sign qubit, swaps the data register with a |0> ancilla register when
x < 0, leaving ReLU(x) = max(0, x) in the data register.
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

# --------------------------------------------------------------------------
# Sigmoid via geometric-series circuit (same corrected construction)
# --------------------------------------------------------------------------
def geometric_state(r, n_qubits):
    for k in range(n_qubits):
        a = 0.5 * 2**k
        theta = 2 * np.arcsin(r**a / np.sqrt(1 + r**(2 * a)))
        qml.RY(theta, wires=k)


def sigmoid_quantum(x, n_qubits, shots):
    if x < 0:
        return 1.0 - sigmoid_quantum(-x, n_qubits, shots)
    r = np.exp(-x)
    dev = qml.device('default.qubit', wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit():
        geometric_state(r, n_qubits)
        return qml.expval(qml.PauliZ(0))

    Z0 = circuit()
    # exact single-qubit reduction: sigma(x) = (1 + <Z_0>)/2 (no truncation factor)
    return 0.5 * (1.0 + Z0)


def softplus_quantum(x, n_qubits, shots, T=8.0, n_nodes=120):
    """softplus(x) for x < 0 by trapezoidal quadrature of quantum sigma."""
    assert x < 0
    grid = np.linspace(-T, x, n_nodes)
    vals = np.array([sigmoid_quantum(t, n_qubits, shots) for t in grid])
    return np.trapezoid(vals, grid)


# --------------------------------------------------------------------------
# ReLU via CSWAP
# --------------------------------------------------------------------------
def relu_quantum(x, n_mag, shots=1000):
    """ReLU for signed integer x with n_mag magnitude bits."""
    wires_sign = 0
    wires_data = list(range(1, 1 + n_mag))
    wires_ancilla = list(range(1 + n_mag, 1 + 2 * n_mag))
    total = 1 + 2 * n_mag

    dev = qml.device('default.qubit', wires=total, shots=shots)

    @qml.qnode(dev)
    def circuit():
        mag = x
        if x < 0:
            qml.PauliX(wires=wires_sign)
            mag = -x
        for i in range(n_mag):
            if (mag >> i) & 1:
                qml.PauliX(wires=wires_data[i])
        for d, a in zip(wires_data, wires_ancilla):
            qml.CSWAP(wires=[wires_sign, d, a])
        return qml.probs(wires=wires_data)

    probs = circuit()
    outcome = int(np.argmax(probs))
    value = 0
    for i in range(n_mag):
        bit = (outcome >> (n_mag - 1 - i)) & 1
        value += bit * (2**i)
    return value


if __name__ == "__main__":
    n_qubits = 5
    shots = 10000

    # ---- Softplus ----
    x_neg = np.linspace(-5.0, -0.25, 25)
    soft_q = np.array([softplus_quantum(x, n_qubits, shots) for x in x_neg])
    # extend to positive side via softplus(x) = x + softplus(-x)
    soft_q_pos = -x_neg + soft_q[::-1] * 0  # placeholder, computed below
    x_pos = -x_neg[::-1]
    soft_q_pos = x_pos + soft_q[::-1]
    x_all = np.concatenate([x_neg, x_pos])
    soft_all = np.concatenate([soft_q, soft_q_pos])
    soft_exact = np.log1p(np.exp(x_all))

    # ---- ReLU ----
    n_mag = 3                      # 1 sign + 3 magnitude bits -> range -8..7
    x_ints = np.arange(-8, 8)
    relu_q = [relu_quantum(x, n_mag) for x in x_ints]
    relu_exact = np.maximum(0, x_ints)

    plt.figure(figsize=(14, 5))
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.grid'] = True

    plt.subplot(1, 2, 1)
    plt.plot(x_all, soft_exact, 'b-', label='Exact softplus')
    plt.plot(x_all, soft_all, 'r--', label=f'Quantum quadrature (n={n_qubits}, {shots} shots/node)')
    plt.xlabel('x'); plt.ylabel(r'$\zeta(x)$'); plt.legend()
    plt.title('Softplus - quadrature of quantum sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(x_ints, relu_exact, 'b-o', label='Exact ReLU')
    plt.plot(x_ints, relu_q, 'r--s', label='Quantum ReLU (CSWAP)')
    plt.xlabel('x'); plt.ylabel('ReLU(x)'); plt.legend()
    plt.title('ReLU - conditional swap circuit')

    plt.tight_layout()
    plt.savefig('softplus_relu_simulation.png', dpi=150)

    print("softplus max err:", np.max(np.abs(soft_all - soft_exact)))
    print("relu exact match:", all(int(a) == int(b) for a, b in zip(relu_q, relu_exact)))
