"""
Quantum activation functions via geometric generating functions - exact estimator.

After the revision, the sigmoid circuit uses the EXACT single-qubit reduction:
the geometric product state with amplitudes ∝ r^{j/2} (r = e^{-x}) has
    <Z_0> = (1 - r)/(1 + r) = tanh(x/2)   exactly, for every n >= 1,
so sigma(x) = (1 + <Z_0>)/2 with NO truncation error, and tanh(x) = <Z_0>
directly with r = e^{-2x}. The remaining n-1 qubits do not affect the Z_0
readout (verified below), and the only error source is shot noise.

Panels:
(a) sigmoid via (1 + <Z_0>)/2, n=5 qubits, 10000 shots per point
(b) tanh via <Z_0> with r = e^{-2x}, n=5 qubits, 10000 shots per point
(c) RMS estimation error vs number of shots M for n in {1,3,5} at x=1,
    showing the 1/sqrt(M) shot-noise scaling and exact n-independence.
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def geometric_angles(r, n_qubits):
    """theta_k = 2 arcsin(r^{2^{k-1}} / sqrt(1 + r^{2^k})): amplitudes ∝ r^{j/2}."""
    return [2 * np.arcsin(r**(2**(k - 1)) / np.sqrt(1 + r**(2**k)))
            for k in range(n_qubits)]


def measure_Z0(x, n_qubits, shots, scale=1.0):
    """<Z_0> on the geometric state with r = exp(-scale * x)."""
    r = np.exp(-scale * x)
    angles = geometric_angles(r, n_qubits)
    dev = qml.device('default.qubit', wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit():
        for k in range(n_qubits):
            qml.RY(angles[k], wires=k)
        return qml.expval(qml.PauliZ(0))

    return circuit()


if __name__ == "__main__":
    n_qubits = 5
    shots = 10000
    xs = np.linspace(0.05, 6.0, 40)

    # Panel (a): sigmoid, x > 0 (x < 0 via sigma(x) = 1 - sigma(-x))
    sig_q = [0.5 * (1 + measure_Z0(x, n_qubits, shots)) for x in xs]
    sig_e = 1 / (1 + np.exp(-xs))

    # Panel (b): tanh directly, x > 0
    tanh_q = [measure_Z0(x, n_qubits, shots, scale=2.0) for x in xs]
    tanh_e = np.tanh(xs)

    # Panel (c): RMS error vs shots at x = 1 for n in {1, 3, 5}
    M_list = [100, 400, 1600, 6400, 25600]
    reps = 200
    x0 = 1.0
    sig0 = 1 / (1 + np.exp(-x0))
    rms = {}
    for n in [1, 3, 5]:
        errs = []
        for M in M_list:
            est = np.array([0.5 * (1 + measure_Z0(x0, n, M)) for _ in range(reps)])
            errs.append(np.sqrt(np.mean((est - sig0)**2)))
        rms[n] = errs
    # analytic shot-noise reference: std of (1+Z)/2 estimator
    r0 = np.exp(-x0)
    Z0 = (1 - r0) / (1 + r0)
    ref = np.sqrt(1 - Z0**2) / (2 * np.sqrt(np.array(M_list)))

    # ---- Plot ----
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.grid'] = True
    plt.figure(figsize=(17, 5))

    plt.subplot(1, 3, 1)
    plt.plot(xs, sig_e, 'b-', label='Exact sigmoid')
    plt.plot(xs, sig_q, 'r--o', ms=4, label=f'Quantum est. ({shots} shots)')
    plt.xlabel('x'); plt.ylabel(r'$\sigma(x)$'); plt.legend()
    plt.title(r'Sigmoid: $\sigma(x) = (1+\langle Z_0\rangle)/2$ (exact)')

    plt.subplot(1, 3, 2)
    plt.plot(xs, tanh_e, 'b-', label='Exact tanh')
    plt.plot(xs, tanh_q, 'r--o', ms=4, label=f'Quantum est. ({shots} shots)')
    plt.xlabel('x'); plt.ylabel(r'$\tanh(x)$'); plt.legend()
    plt.title(r'tanh$(x) = \langle Z_0\rangle$ with $r=e^{-2x}$ (exact)')

    plt.subplot(1, 3, 3)
    for n, sty in zip([1, 3, 5], ['-o', '--s', '-.^']):
        plt.loglog(M_list, rms[n], sty, ms=5, label=f'n={n} qubits')
    plt.loglog(M_list, ref, 'k:', label=r'analytic $\propto 1/\sqrt{M}$')
    plt.xlabel('shots $M$'); plt.ylabel('RMS error'); plt.legend(fontsize=10)
    plt.title(r'Shot-noise scaling at $x=1$ (n-independent)')

    plt.tight_layout()
    plt.savefig('geometric_activation_functions.png', dpi=150)
    print('panel (a) max |sig_q - sig_e|:', np.max(np.abs(np.array(sig_q) - sig_e)))
    print('panel (b) max |tanh_q - tanh_e|:', np.max(np.abs(np.array(tanh_q) - tanh_e)))
    for n in [1, 3, 5]:
        print(f'panel (c) n={n} RMS errors:', np.round(rms[n], 5).tolist())
