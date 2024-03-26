from dataset import DigitsDataSet
import torch
import os
from torch import nn

device = "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8*8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(8*8, 10)
        )
    def forward(self, x):
        x = self.flatten(x.to(torch.float32))
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    X = torch.rand(1, 8, 8, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(y_pred)