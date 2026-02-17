import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Softplus – approximate product state (as in the paper)
# ============================================================================
def softplus_circuit(x, n_qubits):
    r"""Prepare the approximate state |ζ(x)⟩ given in the paper.

    For each qubit k, angle φ_k = 2 arcsin( exp(2^k x) / sqrt(2^(k+1)) ).
    This state is claimed to approximate the softplus series for x<0.
    """
    for k in range(n_qubits):
        arg = np.exp(2**k * x) / np.sqrt(2**(k+1))
        # Clip to valid domain for arcsin (numerical safety)
        if arg > 1.0:
            arg = 1.0
        if arg < -1.0:
            arg = -1.0
        phi = 2 * np.arcsin(arg)
        qml.RY(phi, wires=k)

def softplus_quantum(x, n_qubits, shots=10000):
    """Return the expectation ⟨Z₀⟩ of the state prepared by softplus_circuit.

    The paper does not specify how to extract the softplus value from the state;
    here we simply measure Z on the first qubit for illustration.
    """
    if x >= 0:
        # The series representation is only valid for x<0; for x≥0 we return NaN
        return np.nan

    dev = qml.device('default.qubit', wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit():
        softplus_circuit(x, n_qubits)
        return qml.expval(qml.PauliZ(0))

    return circuit()

# ============================================================================
# ReLU – corrected conditional circuit (single CSWAP)
# ============================================================================
def relu_circuit(sign_wire, data_wires, ancilla_wires):
    """Correct ReLU implementation using a single CSWAP per data qubit.

    If sign_wire = 1 (negative input), swap each data qubit with its ancilla
    (which is initially |0⟩), thereby setting the data register to zero.
    If sign_wire = 0, data remains unchanged.
    """
    for d, a in zip(data_wires, ancilla_wires):
        qml.CSWAP(wires=[sign_wire, d, a])

def relu_quantum(x, n_data_qubits, shots=10000):
    """Simulate ReLU for a signed integer x with n_data_qubits bits (including sign).

    The integer x must lie in the range [-2^(n_data_qubits-1), 2^(n_data_qubits-1)-1].
    """
    n_mag = n_data_qubits - 1                      # number of magnitude qubits
    total_qubits = 1 + n_mag + n_mag                # sign + data + ancilla
    wires_sign = 0
    # Data qubits: we will encode them with qubit 1 as LSB, qubit 2 as next, etc.
    # But PennyLane's probability ordering treats the first wire in the list as MSB.
    # To avoid confusion, we keep the same order in encoding and decoding.
    wires_data = list(range(1, 1 + n_mag))          # qubits 1..n_mag (qubit 1 = LSB)
    wires_ancilla = list(range(1 + n_mag, 1 + 2 * n_mag))

    dev = qml.device('default.qubit', wires=total_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit():
        # Encode x in basis: sign qubit, data qubits for magnitude
        sign = 0
        mag = x
        if x < 0:
            sign = 1
            mag = -x
        if sign:
            qml.PauliX(wires=wires_sign)
        # Encode magnitude in data qubits (binary, LSB in lowest-numbered qubit)
        for i in range(n_mag):
            if (mag >> i) & 1:
                qml.PauliX(wires=wires_data[i])
        # Ancilla start in |0⟩ automatically
        # Apply corrected ReLU circuit
        relu_circuit(wires_sign, wires_data, wires_ancilla)
        # Measure the data register – now contains ReLU(x)
        return qml.probs(wires=wires_data)

    probs = circuit()
    outcome = np.argmax(probs)   # most probable basis state

    # Decode the outcome correctly: because wires_data are ordered [1,2,3] with qubit 1 as LSB,
    # the index 'outcome' has qubit 1 as the most significant bit in its binary representation.
    # We need to map bits back to their original weights.
    value = 0
    for i in range(n_mag):
        # i = 0 corresponds to the least significant bit in our encoding (qubit 1)
        # In the outcome, that bit is at position (n_mag-1-i)
        bit = (outcome >> (n_mag - 1 - i)) & 1
        value += bit * (2**i)
    return value

# ============================================================================
# Numerical experiments
# ============================================================================

# ---- Softplus ----
n_soft = 5
shots_soft = 5000
x_neg = np.linspace(-5.0, -0.1, 50)
soft_quant = []
soft_exact = []
for x in x_neg:
    soft_quant.append(1-softplus_quantum(x, n_soft, shots_soft))
    soft_exact.append(np.log(1 + np.exp(x)))

plt.figure(figsize=(15, 5))
plt.rcParams.update({'font.size': 14})
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.subplot(1,2,1)
plt.plot(x_neg, soft_exact, 'b-', label='Exact softplus')
plt.plot(x_neg, soft_quant, 'r--', label=f'Quantum ⟨Z₀⟩ (n={n_soft}, shots={shots_soft})')
plt.xlabel('x')
plt.ylabel('ζ(x)')
plt.legend()
plt.grid(True)
plt.title('Softplus')
# ---- ReLU (corrected with proper decoding) ----
n_data = 4   # 1 sign + 3 magnitude → range -8..7
x_ints = np.arange(-8, 8)
relu_quant = []
relu_classic = [max(0, x) for x in x_ints]
for x in x_ints:
    relu_quant.append(relu_quantum(x, n_data, shots=1000))

plt.subplot(1,2,2)
plt.plot(x_ints, relu_classic, 'b-o', label='Exact ReLU')
plt.plot(x_ints, relu_quant, 'r--s', label='Quantum (corrected circuit)')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.legend()
plt.grid(True)
plt.title('ReLU – single‑CSWAP circuit')

plt.tight_layout()
plt.savefig('softplus_relu_simulation.png', dpi=150)
plt.show()

# Print values
print("\nSoftplus (⟨Z₀⟩) – not equal to softplus:")
print("x\tExact\t⟨Z₀⟩")
for i in range(0, len(x_neg), 10):
    print(f"{x_neg[i]:.2f}\t{soft_exact[i]:.6f}\t{soft_quant[i]:.6f}")

print("\nReLU (corrected circuit):")
print("x\tExact\tQuantum")
for i, x in enumerate(x_ints):
    print(f"{x:2d}\t{relu_classic[i]:2d}\t{relu_quant[i]:2d}")