"""
Multi‑Fourier Quantum KAN Layer
================================
Combines multiple Fourier terms (each with its own learnable frequency) via a linear layer.
All tensors use float32 for consistency.
"""

import torch
import pennylane as qml
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# FourierTerm – single frequency
# ----------------------------------------------------------------------
class FourierTerm(torch.nn.Module):
    """
    Single Fourier term: returns Re( (1/N) Σ e^{i ω k x} )
    """
    def __init__(self, n_qubits=5, shots=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        self.shots = shots
        self.omega = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.dev = qml.device('default.qubit', wires=n_qubits + 1, shots=shots)

        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def circuit_real(phase):
            qml.Hadamard(wires=0)
            for k in range(1, self.n_qubits + 1):
                qml.Hadamard(wires=k)
            for k in range(self.n_qubits):
                angle = phase * (2 ** k)
                qml.ControlledPhaseShift(angle, wires=[0, k + 1])
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def circuit_imag(phase):
            qml.Hadamard(wires=0)
            for k in range(1, self.n_qubits + 1):
                qml.Hadamard(wires=k)
            for k in range(self.n_qubits):
                angle = phase * (2 ** k)
                qml.ControlledPhaseShift(angle, wires=[0, k + 1])
            qml.adjoint(qml.S)(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        self.circuit_real = circuit_real
        self.circuit_imag = circuit_imag

    def forward(self, x):
        results = []
        for xi in x:
            phase = self.omega * xi
            re = self.circuit_real(phase)
            im = self.circuit_imag(phase)
            avg = re + 1j * im
            results.append(avg.real.float())   # ensure float32
        return torch.stack(results)


# ----------------------------------------------------------------------
# MultiFourierEncoding – linear combination of several FourierTerm
# ----------------------------------------------------------------------
class MultiFourierEncoding(torch.nn.Module):
    """
    Linear combination of multiple Fourier terms.
    Args:
        n_terms (int): number of Fourier terms
        n_qubits (int): number of qubits per term
        shots (int or None): shots for each quantum circuit
    """
    def __init__(self, n_terms=4, n_qubits=5, shots=None):
        super().__init__()
        self.n_terms = n_terms
        self.n_qubits = n_qubits
        self.shots = shots

        # Create a list of FourierTerm modules
        self.terms = torch.nn.ModuleList([FourierTerm(n_qubits, shots) for _ in range(n_terms)])

        # Learnable weights and bias – float32
        self.weights = torch.nn.Parameter(torch.randn(n_terms, dtype=torch.float32) * 0.1)
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        """
        x: tensor of shape (batch,)
        Returns: tensor of shape (batch,)
        """
        term_outputs = [term(x) for term in self.terms]   # list of (batch,) tensors
        stacked = torch.stack(term_outputs, dim=1)        # (batch, n_terms)
        out = torch.mv(stacked, self.weights) + self.bias
        return out


# ----------------------------------------------------------------------
# Quantum KAN Layer (generic)
# ----------------------------------------------------------------------
class QuantumKANLayer(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, univariate_fn_constructor, activation=torch.nn.Identity()):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation

        self.fns = torch.nn.ModuleList([
            torch.nn.ModuleList([univariate_fn_constructor() for _ in range(n_inputs)])
            for _ in range(n_outputs)
        ])

    def forward(self, x):
        batch = x.shape[0]
        S = torch.zeros(batch, self.n_outputs, device=x.device, dtype=torch.float32)
        for j in range(self.n_outputs):
            for i in range(self.n_inputs):
                val = self.fns[j][i](x[:, i])
                S[:, j] += val
        return self.activation(S)


# ----------------------------------------------------------------------
# Training function (unchanged)
# ----------------------------------------------------------------------
def train_kan(model, X_train, y_train, X_val, y_val,
              epochs=100, lr=0.01, batch_size=32, loss_fn=torch.nn.MSELoss(),
              optimizer_class=torch.optim.Adam, verbose=True):
    optimizer = optimizer_class(model.parameters(), lr=lr)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * X_batch.shape[0]

        train_loss = total_loss / len(X_train)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = loss_fn(y_val_pred, y_val).item()
            val_losses.append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            param_str = ', '.join([f'{name[0]}: {param.data.abs().mean():.3f}' for name, param in zip(model.named_parameters(), model.parameters())])
            print(f"Epoch {epoch+1}/{epochs}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}  Params (mean abs): {param_str}")

    return train_losses, val_losses


# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------
# if __name__ == "__main__":
# Generate dataset: y = sin(x) on [0, π]
X = torch.linspace(0, np.pi, 500).reshape(-1, 1).float()
y = torch.sin(X).float()

# Normalise (helps training)
X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()
X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

# Convert to float32 (already)
X_norm = X_norm.float()
y_norm = y_norm.float()

# Train/val/test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(X_norm, y_norm, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Use MultiFourierEncoding with 4 terms
fn_constructor = lambda: MultiFourierEncoding(n_terms=4, n_qubits=5, shots=None)

model = QuantumKANLayer(n_inputs=1, n_outputs=1,
                        univariate_fn_constructor=fn_constructor,
                        activation=torch.nn.Identity())

# Train
train_losses, val_losses = train_kan(model, X_train, y_train, X_val, y_val,
                                        epochs=50, lr=0.05, batch_size=32)

# Evaluate on test set
model.eval()
with torch.no_grad():
    y_test_pred_norm = model(X_test)
    y_test_pred = y_test_pred_norm * y_std + y_mean
    y_test_true = y_test * y_std + y_mean
    test_mse = torch.nn.functional.mse_loss(y_test_pred, y_test_true)
    print(f"\nTest MSE (denormalized): {test_mse.item():.6f}")

plt.figure(figsize=(15, 5))
plt.rcParams.update({'font.size': 14})
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Curve')

plt.subplot(1,2,2)
sort_idx = X_test.squeeze().argsort()
plt.plot(X_test[sort_idx].numpy(), y_test_true[sort_idx].numpy(), 'o', label='True')
plt.plot(X_test[sort_idx].numpy(), y_test_pred[sort_idx].numpy(), 'x', label='Predicted')
plt.xlabel('x (normalized)')
plt.ylabel('y')
plt.legend()
plt.title('Quantum KAN Approximation (Multi‑Fourier)')
plt.tight_layout()
plt.savefig('quantum_kan_multi_fourier.png')
plt.show()