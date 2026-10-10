# quantum-generating-functions
repo for the paper: Quantum Generating Functions: Formulation and Applications in Quantum Machine Learning, 

links to the paper:
[https://doi.org/10.5281/zenodo.18410033](https://doi.org/10.5281/zenodo.18410033)  
[https://www.researchgate.net/publication/400516094_Quantum_Generating_Functions_Formulation_and_Applications_in_Quantum_Machine_Learning](https://www.researchgate.net/publication/400516094_Quantum_Generating_Functions_Formulation_and_Applications_in_Quantum_Machine_Learning)


## Contents

- `paper/` : LaTeX sources of the paper (`main.tex`, `main.bib`, `picture.tex`). Build with `cd paper && latexmk -pdf main.tex`; figures are read from the repository root.
- `sigmoid_tanh.py` : sigmoid / tanh circuits (figure `geometric_activation_functions.png`)
- `soft_plus_relu.py` : softplus by quadrature of the sigmoid circuit, ReLU by CSWAP (figure `softplus_relu_simulation.png`)
- `kernel_simulation.py` : geometric and product-state Gaussian kernels, kernel ridge regression (figure `kernel_simulation.png`)
- `quantum_kan.py` : quantum KAN layer with Fourier edges (figure `quantum_kan_multi_fourier.png`)

Install dependencies with `pip install -r requirements.txt`, then run each script from the repository root.
