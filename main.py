import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import transformed_dataset
from model import NeuralNetwork
from torch.utils.data import random_split
from math import ceil
from evolutionary import evolve


dataset = random_split(
    transformed_dataset, (898, 899)
)  # the tuple elements have to sum up to 1797
train_dataset = dataset[0]
test_dataset = dataset[1]

model = NeuralNetwork()

# learning rate - describes the size of the step taken by our network (step size = gradient * learning rate)
learning_rate = 30e-3
# batch size - the number of data points from which the gradient will be calculated each iteration
batch_size = 40
epochs = 20


train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)


loss_fn = nn.CrossEntropyLoss()
# SGD optimizer - implements stochastic gradient descent,
# which means selecting a group of random data points and
# performing calculations on those. Kind of like another batch size? Unsure.

# after plugging in momentum, the model more quickly converged onto a local minimum,
# but the accuracy was worse (93.2% to 95.3%)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # doesn't train the model, simply puts it in training mode. Some
    # things in model work differently depending on train/eval mode.
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # makes model give predictions based on its current parameters
        pred = model(X)
        # calculates loss
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * batch_size + len(y)
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


def load_params_to_model(model, params):
    pretrained_dict = model.state_dict()
    counter = 0

    for key in pretrained_dict.keys():
        pretrained_dict[key] = torch.from_numpy(params[counter])
        counter += 1

    model.load_state_dict(pretrained_dict)

if __name__ == "__main__":
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train_loop(train_dataloader, model, loss_fn, optimizer)
    # print("Done!")

    # torch.save(model, "og_model.txt")

    # model2 = torch.load('og_model.txt')

    load_params_to_model(model, evolve(50, 100, train_dataloader))

    print("\n Testing")
    test_loop(test_dataloader, model, loss_fn)

    # load_params_to_model(model, create_param_sets(1)[0])

    # print("\n Testing")
    # test_loop(test_dataloader, model, loss_fn)