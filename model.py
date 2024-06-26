import torch
from torch import nn

# this could work better (device agnostic code)
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(8 * 8, 10),
        )

    def forward(self, x):
        # sklearn used float64, idk if it should be converted in this place but it works
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
