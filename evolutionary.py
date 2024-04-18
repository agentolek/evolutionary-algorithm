import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from dataset import transformed_dataset
from model import NeuralNetwork
from torch.utils.data import random_split

model = NeuralNetwork()
parameters = [[name, p] for name, p in model.named_parameters()]

names = [name for name, p in parameters]

def create_random_tensor(*args):
    array = np.random.rand(*args) / 10
    # TODO: add negative numbers to random arrays
    # for _ in range(array.size * 0.5):
    return array

params_sets = []
for _ in range(100):
    tmp = []
    tmp.append(np.random.rand(64, 64) / 10)
    tmp.append(np.random.rand(64) / 10)
    tmp.append(np.random.rand(64, 64) / 10)
    tmp.append(np.random.rand(64) / 10)    
    tmp.append(np.random.rand(64, 64) / 10)
    tmp.append(np.random.rand(64) / 10)
    tmp.append(np.random.rand(10, 64) / 10)
    tmp.append(np.random.rand(10) / 10)
    params_sets.append(tmp)


def load_params_to_model(params):
    pretrained_dict = model.state_dict()
    counter = 0

    for key in pretrained_dict.keys():
        pretrained_dict[key] = torch.from_numpy(params[counter])
        counter += 1

    model.load_state_dict(pretrained_dict)
    parameters = [[name, p] for name, p in model.named_parameters()]


def eval_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    return test_loss


dataset = random_split(
    transformed_dataset, (898, 899)
)  # the tuple elements have to sum up to 1797
train_dataset = dataset[0]
batch_size = 40

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

def evaluate(params):
    load_params_to_model(params)
    score = eval_loop(train_dataloader, model, nn.CrossEntropyLoss())
    return score


def evaluate_all(param_list):
    scores = []
    for elem in param_list:
        scores.append(evaluate(elem))
    return [x / -100 for x in scores]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# uses random resetting
def mutate(array, mutation_prob):

    mask = np.random.randint(0,1/mutation_prob,size=array.shape).astype(bool)
    print(mask)
    r = (np.random.rand(*array.shape) - 0.5) * 0.4
    print(r)
    array[mask] = r[mask]
    return array


prob_distribution = softmax(np.array(evaluate_all(params_sets)))
print(prob_distribution)
selected_index = np.random.choice(np.arange(100), size=50, p=prob_distribution, replace=False)
selected_params = [params_sets[x] for x in selected_index]