"""
Model classes for linear probe evaluation.
"""
import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Single linear layer classifier for linear probe evaluation."""
    
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class MLPClassifier(nn.Module):
    """Multi-layer perceptron classifier for more complex evaluation."""
    
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.mlp(x)
