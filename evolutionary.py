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

def create_param_sets(gen_size):
    sets = []
    for _ in range(gen_size):
        tmp = []
        tmp.append(np.random.rand(64, 64) / 10)
        tmp.append(np.random.rand(64) / 10)
        tmp.append(np.random.rand(64, 64) / 10)
        tmp.append(np.random.rand(64) / 10)    
        tmp.append(np.random.rand(64, 64) / 10)
        tmp.append(np.random.rand(64) / 10)
        tmp.append(np.random.rand(10, 64) / 10)
        tmp.append(np.random.rand(10) / 10)
        sets.append(tmp)
    
    return sets


def load_params_to_model(params):
    pretrained_dict = model.state_dict()
    counter = 0

    for key in pretrained_dict.keys():
        pretrained_dict[key] = torch.from_numpy(params[counter])
        counter += 1

    model.load_state_dict(pretrained_dict)


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
    return [x / -0.1 for x in scores]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# uses random resetting
def mutate(array, mutation_prob):

    mask = np.random.randint(0,1/mutation_prob,size=array.shape).astype(bool)
    # currently generates random values in the range <-0.1, 0.1>
    r = (np.random.rand(*array.shape) - 0.5) * 0.2
    array[mask] = r[mask]
    return array

def mutate_set(params_set, mutation_probs):
    new_set = []
    for i in range(len(params_set)):
        new_set.append(mutate(params_set[i], mutation_probs[i]))
    return new_set

def cross_breed(pair):
    mother, father = pair
    split = np.random.randint(1, len(father))

    child1 = mother[:split] + father[split:]
    child2 = father[:split] + mother[split:]
    
    return child1, child2

def choose_parent_indexes(num_of_pairs, index_range):
    # num of pairs - how many pairs to create
    # index_range - what indexes to select from
    pairs = []
    for index in range(num_of_pairs):
        pairs.append([index, np.random.randint(index_range)])
    return pairs


def create_generation(params_sets, gen_size, layer_mutate_prob):
    
    pairs = choose_parent_indexes(gen_size // 2, gen_size // 2)
    new_gen = []

    for pair in pairs:
        # this for loop creates kids from pairs of params
        pair = [params_sets[x] for x in pair]
        new_gen += cross_breed(pair)
    
    for i in range(gen_size):
        # this for loop is responsible for mutations in kids
        new_gen[i] = mutate_set(new_gen[i], layer_mutate_prob)

    return tuple(new_gen)


def evolve(epochs, gen_size):

    params_sets = create_param_sets(gen_size) # begin with random arrays of parameters
    for _ in range(epochs):
        prob_distribution = softmax(np.array(evaluate_all(params_sets)))
        # print(evaluate_all(params_sets)) 
        # print(prob_distribution)
        # print("\n")
        selected_index = np.random.choice(np.arange(gen_size), size=gen_size//2, p=prob_distribution, replace=False)
        selected_params = [params_sets[x] for x in selected_index]
        params_sets = create_generation(selected_params, gen_size, layer_mutate_prob=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008])


    temp = list(zip(evaluate_all(params_sets), params_sets))
    sorted(temp, key=lambda x: x[0])
    
    return temp[0][1]
    
if __name__ == "__main__":
    random_params = create_param_sets(1)
    params = evolve(200, 100)
    print("Random loss: " + str(evaluate(params)))
    print("Evolved loss: " + str(evaluate(params)))
    