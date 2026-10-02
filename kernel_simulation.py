"""
Quantum kernel methods via generating functions - numerical demonstration.

Panel (a): Geometric kernel. The product state
    |phi(x)> = tensor_k (|0> + x^{2^k}|1>) / sqrt(1 + x^{2^{k+1}})
has amplitudes proportional to x^j, so
    <phi(x)|phi(y)> = sum_{j=0}^{N-1} (xy)^j / sqrt(S_N(x) S_N(y)),
    S_N(x) = (1 - x^{2N})/(1 - x^2),
i.e. the (normalised) truncated geometric kernel. Estimated by SWAP test.

Panel (b): Product-state Gaussian kernel. d identical qubits in
    |u(x)> = sqrt(1 - (gx/sqrt(d))^2)|0> + (gx/sqrt(d))|1>
give |<phi(x)|phi(y)>|^2 = [sqrt((1-qx^2)(1-qy^2)) + qx qy]^{2d}
    -> exp(-g^2 (x - y)^2)  as d -> infinity, with error O(1/d).
Estimated by SWAP test on 2d+1 qubits.

Panel (c): Kernel ridge regression with the quantum geometric kernel
(SWAP-test kernel matrix) on a 1-D binary classification task.
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

# --------------------------------------------------------------------------
# Geometric kernel
# --------------------------------------------------------------------------
def geometric_feature_angles(x, n_qubits):
    return [2 * np.arcsin(x**(2**k) / np.sqrt(1 + x**(2**(k + 1))))
            for k in range(n_qubits)]


def swap_test_overlap_sq(prep_x, prep_y, n, shots):
    """| <phi(x)|phi(y) > |^2 via SWAP test; prep_* act on wires 1..n / n+1..2n."""
    dev = qml.device('default.qubit', wires=2 * n + 1, shots=shots)

    @qml.qnode(dev)
    def circuit():
        prep_x(range(1, n + 1))
        prep_y(range(n + 1, 2 * n + 1))
        qml.Hadamard(wires=0)
        for i in range(n):
            qml.CSWAP(wires=[0, 1 + i, n + 1 + i])
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    return circuit()   # <Z> = |<phi(x)|phi(y)>|^2


def geo_kernel_quantum(x, y, n_qubits, shots):
    ax = geometric_feature_angles(x, n_qubits)
    ay = geometric_feature_angles(y, n_qubits)
    prep_x = lambda wires: [qml.RY(ax[k], wires=wires[k]) for k in range(n_qubits)]
    prep_y = lambda wires: [qml.RY(ay[k], wires=wires[k]) for k in range(n_qubits)]
    return swap_test_overlap_sq(prep_x, prep_y, n_qubits, shots)


def geo_kernel_exact(x, y, n_qubits):
    N = 2**n_qubits
    j = np.arange(N)
    num = np.sum((x * y)**j)
    Sx = (1 - x**(2 * N)) / (1 - x**2)
    Sy = (1 - y**(2 * N)) / (1 - y**2)
    return (num / np.sqrt(Sx * Sy))**2


# --------------------------------------------------------------------------
# Product-state Gaussian kernel
# --------------------------------------------------------------------------
def rbf_prod_kernel_quantum(x, y, gamma, d, shots):
    q = lambda v: gamma * v / np.sqrt(d)
    def prep(v, wires):
        angle = 2 * np.arcsin(q(v))
        for w in wires:
            qml.RY(angle, wires=w)
    return swap_test_overlap_sq(lambda ws: prep(x, ws), lambda ws: prep(y, ws), d, shots)


def rbf_prod_kernel_exact(x, y, gamma, d):
    qx, qy = gamma * x / np.sqrt(d), gamma * y / np.sqrt(d)
    return (np.sqrt((1 - qx**2) * (1 - qy**2)) + qx * qy)**(2 * d)


if __name__ == "__main__":
    n_qubits = 5
    shots = 20000

    # ---- Panel (a): geometric kernel K(x0, y) ----
    x0 = 0.5
    ys = np.linspace(-0.9, 0.9, 25)
    K_q = [geo_kernel_quantum(x0, y, n_qubits, shots) for y in ys]
    K_e = [geo_kernel_exact(x0, y, n_qubits) for y in ys]

    # ---- Panel (b): product-state Gaussian kernel ----
    # domain constraint: |gamma * x| <= sqrt(d) (single-qubit amplitude <= 1)
    gamma = 1.2
    yb = np.linspace(-1.6, 1.6, 60)
    xb0 = 0.6
    ds = [4, 16, 64]
    Kb = {d: [rbf_prod_kernel_exact(xb0, y, gamma, d) for y in yb] for d in ds}
    Kb_true = np.exp(-gamma**2 * (xb0 - yb)**2)
    # a few SWAP-test estimates with shots for d=8 (17 qubits)
    yb_q = np.linspace(-1.6, 1.6, 9)
    Kb_q = [rbf_prod_kernel_quantum(xb0, y, gamma, 8, shots) for y in yb_q]
    # convergence in d
    d_scan = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    err_d = [abs(rbf_prod_kernel_exact(xb0, 1.0, gamma, d)
                 - np.exp(-gamma**2 * (xb0 - 1.0)**2)) for d in d_scan]

    # ---- Panel (c): kernel ridge regression, multi-seed + classical baseline ----
    n_train = 36
    lam = 0.1
    Xte = np.linspace(0.05, np.pi - 0.05, 200)
    yte = np.sign(np.sin(3 * Xte))

    def swap_noise(K_exact, M, rng):
        """Binomial shot noise statistically identical to an M-shot SWAP test."""
        p = np.clip((1 + K_exact) / 2, 0, 1)
        return 2 * rng.binomial(M, p) / M - 1

    def ridge_acc(Kmat_tr, ytr, Xtr_k, Xte_k, kern_exact):
        alpha = np.linalg.solve(Kmat_tr + lam * np.eye(len(ytr)), ytr)
        Kte = np.array([[kern_exact(xt, xs) for xs in Xtr_k] for xt in Xte_k])
        return np.sign(Kte @ alpha)

    gamma_rbf = 2.0
    Xtr = np.linspace(0.05, np.pi - 0.05, n_train)
    Xn = 0.9 * Xtr / np.pi
    Xte_n = 0.9 * Xte / np.pi
    geo_exact_tr = np.array([[geo_kernel_exact(a, b, n_qubits) for b in Xn] for a in Xn])
    rbf_tr = np.exp(-gamma_rbf**2 * (Xtr[:, None] - Xtr[None, :])**2)
    rbf_te = lambda xt, xs: np.exp(-gamma_rbf**2 * (xt - xs)**2)

    n_seeds = 10
    acc_q, acc_rbf = [], []
    preds = {}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        ytr = np.sign(np.sin(3 * Xtr)); flip = rng.random(n_train) < 0.1; ytr[flip] *= -1
        Kq = swap_noise(geo_exact_tr, shots, rng)
        np.fill_diagonal(Kq, 1.0)  # SWAP test of a state with itself is exactly 1
        pq = ridge_acc(Kq, ytr, Xn, Xte_n, lambda xt, xs: geo_kernel_exact(xt, xs, n_qubits))
        pr = ridge_acc(rbf_tr, ytr, Xtr, Xte, rbf_te)
        acc_q.append(np.mean(pq == yte)); acc_rbf.append(np.mean(pr == yte))
        if seed == 0:
            preds = {'q': pq, 'rbf': pr, 'ytr': ytr}
    acc_q, acc_rbf = np.array(acc_q), np.array(acc_rbf)
    acc, pred = acc_q[0], preds['q']
    ytr = preds['ytr']

    # ---- Plot ----
    plt.figure(figsize=(17, 5))
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.grid'] = True

    plt.subplot(1, 3, 1)
    plt.plot(ys, K_e, 'b-', label='Exact geometric kernel')
    plt.plot(ys, K_q, 'r--o', ms=4, label=f'SWAP test ({shots} shots)')
    plt.xlabel('y'); plt.ylabel(r'$K(0.5,\, y)$'); plt.legend()
    plt.title(f'Geometric kernel (n={n_qubits})')

    plt.subplot(1, 3, 2)
    plt.plot(yb, Kb_true, 'k-', lw=2.5, label=r'Exact $\exp(-\gamma^2(x-y)^2)$')
    for d, sty in zip(ds, ['--', '-.', ':']):
        plt.plot(yb, Kb[d], sty, label=f'Product state, d={d}')
    plt.plot(yb_q, Kb_q, 'rs', ms=5, label=f'SWAP test d=8 ({shots} shots)')
    plt.xlabel('y'); plt.ylabel(r'$K(0.6,\, y)$'); plt.legend(fontsize=10)
    plt.title(r'Product-state Gaussian kernel ($\gamma=1.2$)')

    plt.subplot(1, 3, 3)
    plt.plot(Xte, np.sign(np.sin(3 * Xte)), 'b-', label='True boundary')
    plt.plot(Xte, pred, 'r--', label=f'Quantum geo. kernel ({acc_q.mean():.2f}$\\pm${acc_q.std():.2f})')
    plt.plot(Xte, preds['rbf'], 'g-.', label=f'Classical RBF ({acc_rbf.mean():.2f}$\\pm${acc_rbf.std():.2f})')
    plt.plot(Xtr, ytr, 'ko', ms=4, label='Training data (10% label noise)')
    plt.xlabel('x'); plt.ylabel('class'); plt.legend(fontsize=8.5)
    plt.title('Classification (mean$\\pm$std over 10 seeds)')

    plt.tight_layout()
    plt.savefig('kernel_simulation.png', dpi=150)
    print("panel (a) max |K_q - K_e|:", np.max(np.abs(np.array(K_q) - np.array(K_e))))
    print("panel (b) RBF conv err vs d:", list(zip(d_scan.tolist(), np.round(err_d, 5).tolist())))
    print("panel (c) quantum geo kernel acc:", round(acc_q.mean(),3), "+/-", round(acc_q.std(),3))
    print("panel (c) classical RBF acc:", round(acc_rbf.mean(),3), "+/-", round(acc_rbf.std(),3))
