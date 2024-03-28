import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import transformed_dataset
from model import NeuralNetwork

# uhh i think the example on pytorch.org was bad and i actually have to split the data
# OLEK - data has now been split
train_split = int(0.8 * len(transformed_dataset))
train_dataloader = DataLoader(
    transformed_dataset[:train_split], batch_size=64, shuffle=True
)
test_dataloader = DataLoader(
    transformed_dataset[train_split:], batch_size=64, shuffle=True
)


model = NeuralNetwork()
learning_rate = 1e-3
batch_size = 64
momentum = 0.9
epochs = 30

loss_fn = nn.CrossEntropyLoss()

# research other optimizers later
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# code in docs was prob outdated
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    # print(dataloader[0])
    for index, data in enumerate(dataloader):
        # I do not know what is going on, need to look at shape for dataloader
        inputs, labels = data
        pred = model(inputs)

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
