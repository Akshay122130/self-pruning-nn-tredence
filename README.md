# Self-Pruning Neural Network (Case Study)

This repository implements a **Self-Pruning Neural Network** using PyTorch. The core idea is to learn "gates" for each weight in a linear layer, where an L1 penalty on these gates encourages the network to automatically identify and prune unimportant connections during training.

## Theory: L1 Penalty on Sigmoid Gates

### Why it encourages sparsity
The mechanism uses a learnable parameter called `gate_scores` ($s$) for each weight. The actual gate applied to the weight is $g = \sigma(s)$, where $\sigma$ is the Sigmoid function.

1.  **Gating Mechanism**: The effective weight used in the forward pass is $W_{eff} = W \odot \sigma(s)$.
2.  **Sparsity Loss**: We add an L1 penalty to the total loss: $\mathcal{L}_{sparsity} = \lambda \sum |g| = \lambda \sum \sigma(s)$.
3.  **Gradient Dynamics**: The gradient of the sparsity loss with respect to the gate score $s$ is:
    $$\frac{\partial \mathcal{L}_{sparsity}}{\partial s} = \lambda \cdot \sigma(s)(1 - \sigma(s))$$
4.  **Pressure towards Zero**: This gradient always has the same sign as $\lambda$. It continuously pushes $s$ towards $-\infty$, which drives the gate value $g = \sigma(s)$ towards $0$. 
5.  **Selective Pruning**: During training, the Cross-Entropy loss will provide gradients to keep "important" gates open (near 1) to maintain accuracy. The sparsity penalty will successfully drive "unimportant" gates to zero, effectively pruning those connections without significantly impacting performance.

## Experimental Results (CIFAR-10)

The following results were obtained using a 3-layer MLP on a subset of CIFAR-10 (5000 train, 1000 test images) for 20 epochs with decoupled learning rates (weights: 1e-3, gates: 0.05).

| Lambda ($\lambda$) | Test Accuracy | Sparsity Level % (Gates < 0.01) |
| :--- | :--- | :--- |
| 0.0 (Baseline) | 39.40% | 0.02% |
| 0.005 (Low) | 40.90% | 99.87% |
| 0.01 (Medium) | 36.50% | 99.95% |
| 0.02 (High) | 29.90% | 99.99% |

## Project Structure

- `main.py`: Contains the `PrunableLinear` layer, model architecture, training loop (with decoupled optimizer), and evaluation logic.
- `gate_distribution.png`: Visualization of the gate value distribution for the best performing model.
- `requirements.txt`: List of dependencies.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to perform the experiments and generate the visualization:

```bash
python main.py
```

The script will:
1. Download the CIFAR-10 dataset.
2. Run training for 4 different lambda values using a decoupled optimizer.
3. Print a summary table of results.
4. Save a histogram of the final gate values to `gate_distribution.png`.

## Implementation Details

- **Custom Layer**: `PrunableLinear` implements the gating logic from scratch using `nn.Parameter`.
- **Decoupled Optimization**: The training loop uses an Adam optimizer with two parameter groups:
  - **Standard Weights/Biases**: Learning rate of 1e-3.
  - **Gate Scores**: Learning rate of 0.05 (accelerated to overcome sigmoid gradient vanishing).
- **Metrics**: Sparsity is calculated as the percentage of gate values below a 0.01 threshold.
- **Type Hinting**: The codebase uses Python type hints for better readability and production-grade quality.
