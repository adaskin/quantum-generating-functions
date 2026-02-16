import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Circuit for the sigmoid state (Theorem 4.1)
# ----------------------------------------------------------------------
def sigmoid_circuit(x, n_qubits):
    r"""Prepare the state :math:`|\sigma(x)\rangle` as per the paper.

    For each qubit k = 0,..., n-1 apply an R_y gate with angle
    :math:`\theta_k = 2\arcsin(e^{-(2^k+1)x})`.
    """
    for k in range(n_qubits):
        theta = 2 * np.arcsin(np.exp(-(2**k + 1) * x))
        qml.RY(theta, wires=k)

# ----------------------------------------------------------------------
# Circuit for the tanh state (Theorem 4.2)
# ----------------------------------------------------------------------
def tanh_circuit(x, n_qubits):
    r"""Prepare :math:`|\tanh(x)\rangle` using the scaled sigmoid circuit.

    Angles: :math:`\theta_k = 2\arcsin(e^{-(2^k+1)(2x)})`.
    """
    for k in range(n_qubits):
        theta = 2 * np.arcsin(np.exp(-(2**k + 1) * 2 * x))
        qml.RY(theta, wires=k)

# ----------------------------------------------------------------------
# Set up the quantum device
# ----------------------------------------------------------------------
n_qubits = 5                     # 2^5 = 32 terms in the truncated series
shots = 10000                    # number of measurement shots
dev = qml.device('default.qubit', wires=n_qubits, shots=shots)

# ----------------------------------------------------------------------
# QNodes that return the expectation value of Z on the first qubit
# ----------------------------------------------------------------------
@qml.qnode(dev)
def sigmoid_z_expval(x):
    sigmoid_circuit(x, n_qubits)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def tanh_z_expval(x):
    tanh_circuit(x, n_qubits)
    return qml.expval(qml.PauliZ(0))

# ----------------------------------------------------------------------
# Evaluate over a range of x (only x >= 0; negative x can be handled by symmetry)
# ----------------------------------------------------------------------
x_vals = np.linspace(0.0, 5.0, 50)
sig_quant = [sigmoid_z_expval(x) for x in x_vals]
tanh_quant = [tanh_z_expval(x) for x in x_vals]

# Classical reference functions
sig_classic = 1.0 / (1.0 + np.exp(-x_vals))
tanh_classic = np.tanh(x_vals)

# ----------------------------------------------------------------------
# Plot the results
# ----------------------------------------------------------------------
plt.figure(figsize=(15, 5))
plt.rcParams.update({'font.size': 14})
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 14

plt.subplot(1, 2, 1)
plt.plot(x_vals, sig_classic, 'b-', label='Exact sigmoid')
plt.plot(x_vals, sig_quant, 'r--', label=f'Quantum ⟨Z₀⟩ (n={n_qubits}, shots={shots})')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.title('Sigmoid function')

plt.subplot(1, 2, 2)
plt.plot(x_vals, tanh_classic, 'b-', label='Exact tanh')
plt.plot(x_vals, tanh_quant, 'r--', label=f'Quantum ⟨Z₀⟩ (n={n_qubits}, shots={shots})')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.title('Tanh function')

plt.tight_layout()
plt.savefig('quantum_activation_functions.png', dpi=150)
plt.show()


fig, ax = qml.draw_mpl(tanh_z_expval)(1.2345)
fig.show()