import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Geometric state preparation (Section 3)
# ============================================================================
def geometric_state(r, n_qubits):
    r"""Prepare |ψ_r^(N)⟩ with amplitudes proportional to r^{j/2}.

    For each qubit k, the rotation angle θ_k satisfies
        sin(θ_k/2) = r^{2^{k-1}} / √(1 + r^{2^k})
    with the convention 2^{-1} = 0.5 for k=0.
    """
    for k in range(n_qubits):
        # exponent for this qubit: r^{2^{k-1}}  (k=0 → exponent = 0.5)
        exponent = 2**(k-1) if k > 0 else 0.5
        numerator = r**exponent
        denominator = np.sqrt(1 + r**(2**k))
        theta = 2 * np.arcsin(numerator / denominator)
        qml.RY(theta, wires=k)

# ============================================================================
# Sigmoid via geometric series
# ============================================================================
def sigmoid_geometric(x, n_qubits, shots=10000):
    r"""Estimate σ(x) using the geometric series method (Section 3).

    For x>0: σ(x) = e^{-x} Σ_{j=0}^{∞} (-e^{-x})^j.
    We prepare the state |ψ_r⟩ with r = e^{-x}, apply a Z gate on qubit 0
    (which gives (-1)^j), and extract the truncated series from the expectation.
    """
    if x < 0:
        # symmetry σ(x) = 1 - σ(-x)
        val_pos, _ = sigmoid_geometric(-x, n_qubits, shots)
        return 1.0 - val_pos, 1.0 - val_pos

    r = np.exp(-x)               # modulus
    N = 2**n_qubits

    dev = qml.device('default.qubit', wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit():
        geometric_state(r, n_qubits)
        qml.Z(wires=0)            # implements (-1)^j
        return qml.expval(qml.PauliZ(0))

    expval = circuit()

    # ⟨Z₀⟩ = (1-r)/(1-r^N) Σ_{j=0}^{N-1} (-r)^j
    if r == 1.0:   # x=0
        series_sum = expval * N
    else:
        series_sum = (1 - r**N) / (1 - r) * expval

    sig_est = r * series_sum
    # exact truncated value for comparison
    if r == 1.0:
        sig_exact_trunc = r * N
    else:
        sig_exact_trunc = r * (1 - (-r)**N) / (1 + r)

    return sig_est, sig_exact_trunc

# ============================================================================
# Tanh via geometric series (using tanh(x) = 2σ(2x)-1)
# ============================================================================
def tanh_geometric(x, n_qubits, shots=10000):
    r"""Estimate tanh(x) using tanh(x) = 2σ(2x)-1 and the geometric series method."""
    sig_2x, _ = sigmoid_geometric(2*x, n_qubits, shots)
    return 2 * sig_2x - 1

# ============================================================================
# Draw the circuit for a small example (n=3)
# ============================================================================
n_qubits_draw = 3
x_example = 1.0
r_example = np.exp(-x_example)

dev_draw = qml.device('default.qubit', wires=n_qubits_draw)

@qml.qnode(dev_draw)
def draw_geometric():
    geometric_state(r_example, n_qubits_draw)
    qml.Z(wires=0)
    return qml.state()

print("Geometric state circuit for sigmoid (n=3, x=1.0):")
print(qml.draw(draw_geometric)())

# ============================================================================
# Numerical evaluation over a range of x
# ============================================================================
n_qubits = 5          # 2^5 = 32 terms
shots = 10000
x_vals = np.linspace(0.0, 5.0, 50)

sig_geom = []
tanh_geom = []
sig_exact_trunc_vals = []

for x in x_vals:
    sg, s_trunc = sigmoid_geometric(x, n_qubits, shots)
    sig_geom.append(1-sg)
    sig_exact_trunc_vals.append(s_trunc)
    tanh_geom.append(np.abs(tanh_geometric(x, n_qubits, shots)))

# Classical references
sig_classic = 1.0 / (1.0 + np.exp(-x_vals))
tanh_classic = np.tanh(x_vals)

# ============================================================================
# Plot results
# ============================================================================
plt.figure(figsize=(15, 5))
plt.rcParams.update({'font.size': 14})
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 14

plt.subplot(1, 2, 1)
plt.plot(x_vals, sig_classic, 'b-', label='Exact sigmoid')
plt.plot(x_vals, sig_geom, 'r--', label=f'Geometric method (n={n_qubits}, shots={shots})')
#plt.plot(x_vals, sig_exact_trunc_vals, 'g:', label='Exact truncated series (N=32)')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.legend()
plt.grid(True)
plt.title('Sigmoid – geometric series method')

plt.subplot(1, 2, 2)
plt.plot(x_vals, tanh_classic, 'b-', label='Exact tanh')
plt.plot(x_vals, tanh_geom, 'r--', label=f'Geometric method (n={n_qubits}, shots={shots})')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.legend()
plt.grid(True)
plt.title('Tanh – geometric series method (via sigmoid)')

plt.tight_layout()
plt.savefig('geometric_activation_functions.png', dpi=150)
plt.show()

# Print some values
print("\nSigmoid comparison (geometric method):")
print("x\tExact\tQuantum\tError")
for i in range(0, len(x_vals), 10):
    print(f"{x_vals[i]:.2f}\t{sig_classic[i]:.6f}\t{sig_geom[i]:.6f}\t{abs(sig_classic[i]-sig_geom[i]):.2e}")



