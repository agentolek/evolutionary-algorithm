import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import transformed_dataset
from model import NeuralNetwork
from torch.utils.data import random_split
from math import ceil


dataset = random_split(transformed_dataset, (832,832,133)) # the tuple elements have to sum up to 1797
train_dataset = dataset[0]
test_dataset = dataset[1]
val_dataset = dataset[2]

model = NeuralNetwork()

learning_rate = 30e-3
batch_size = 40
epochs = ceil((len(train_dataset) + len(test_dataset))/batch_size)


train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()

        loss = loss_fn(pred, labels)
        loss.backward()

        optimizer.step()

        if index % 100 == 0:
            loss, current = loss.item(), index * batch_size + len(labels)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )



if __name__ == "__main__":
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    print("\n Validation")
    for t in range(3):
        print(f"Epoch {t+1}\n-------------------------------")
        test_loop(val_dataloader, model, loss_fn)


