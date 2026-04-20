import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
import os


class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable gate scores that can be used for pruning.
    Each weight has an associated gate value (via sigmoid(gate_scores)).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        
        self.gate_scores = nn.Parameter(torch.ones_like(self.weight))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and gate scores."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        
        nn.init.constant_(self.gate_scores, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: apply sigmoid to gate_scores, multiply with weight, 
        then perform linear operation.
        """
        
        gates = torch.sigmoid(self.gate_scores)
        
        
        effective_weight = self.weight * gates
        
        return F.linear(x, effective_weight, self.bias)

    def get_gate_values(self) -> torch.Tensor:
        """Returns the current gate values (after sigmoid)."""
        return torch.sigmoid(self.gate_scores).detach()



class PrunableMLP(nn.Module):
    """Simple Feed-Forward Network for CIFAR-10 using PrunableLinear layers."""
    def __init__(self, input_dim: int = 3072, hidden_dims: List[int] = [128, 64], num_classes: int = 10):
        super(PrunableMLP, self).__init__()
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(PrunableLinear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            curr_dim = h_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = PrunableLinear(curr_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1) 
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_sparsity_loss(self) -> torch.Tensor:
        """Calculates L1 norm of all gate values across all PrunableLinear layers."""
        sparsity_loss = 0.0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                sparsity_loss += torch.sum(torch.sigmoid(module.gate_scores))
        return sparsity_loss

    def get_sparsity_stats(self, threshold: float = 0.01) -> float:
        """Calculates the percentage of weights where the gate value is < threshold."""
        total_weights = 0
        pruned_weights = 0
        
        for module in self.modules():
            
            if hasattr(module, 'gate_scores'):
                
                gates = torch.sigmoid(module.gate_scores).detach()
                total_weights += gates.numel()
                pruned_weights += torch.sum(gates < threshold).item()
        
        return (pruned_weights / total_weights) * 100 if total_weights > 0 else 0.0

    def get_all_gates(self) -> np.ndarray:
        """Collects all gate values into a single numpy array for visualization."""
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gate_values().cpu().numpy().flatten())
        return np.concatenate(all_gates)


def train_and_evaluate(
    model: nn.Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    lambda_sparsity: float, 
    epochs: int = 20,
    device: str = "cpu"
) -> Tuple[float, float]:
    """Trains the model with sparsity penalty and returns test accuracy and final sparsity."""
    
    gate_params = []
    weight_params = []
    
    for name, param in model.named_parameters():
        if 'gate' in name:
            gate_params.append(param)
        else:
            weight_params.append(param)
            
    optimizer = optim.Adam([
        {'params': weight_params, 'lr': 1e-3},
        {'params': gate_params, 'lr': 0.05}
    ])
    
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            ce_loss = criterion(outputs, labels)
            sparsity_loss = model.get_sparsity_loss()
            
            loss = ce_loss + lambda_sparsity * sparsity_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Lambda: {lambda_sparsity} | Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    sparsity_level = model.get_sparsity_stats()
    
    return accuracy, sparsity_level

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    full_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_subset = torch.utils.data.Subset(full_train, range(5000))
    test_subset = torch.utils.data.Subset(full_test, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    lambda_values = [0.0, 0.005, 0.01, 0.02] # Low, Medium, High sparsity pressure
    results = []
    best_model = None
    best_acc = -1
    
    for lmbda in lambda_values:
        print(f"\n--- Running experiment with Lambda = {lmbda} ---")
        model = PrunableMLP().to(device)
        acc, sparsity = train_and_evaluate(model, train_loader, test_loader, lmbda, epochs=20, device=device)
        results.append({
            "lambda": lmbda,
            "accuracy": acc,
            "sparsity": sparsity
        })
        print(f"Results for Lambda {lmbda}: Accuracy: {acc:.2f}%, Sparsity Level: {sparsity:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model

    # --- Results Summary ---
    print("\n" + "="*30)
    print(f"{'Lambda':<10} | {'Test Acc':<10} | {'Sparsity %':<10}")
    print("-" * 35)
    for res in results:
        print(f"{res['lambda']:<10.1e} | {res['accuracy']:<10.2f} | {res['sparsity']:<10.2f}")
    print("="*30)

    # --- Visuals ---
    if best_model:
        all_gates = best_model.get_all_gates()
        plt.figure(figsize=(10, 6))
        plt.hist(all_gates, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f"Distribution of Gate Values (Lambda={results[-1]['lambda']})")
        plt.xlabel("Gate Value (Sigmoid Score)")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig("gate_distribution.png")
        print(f"\nSaved gate distribution plot to gate_distribution.png")

if __name__ == "__main__":
    main()
