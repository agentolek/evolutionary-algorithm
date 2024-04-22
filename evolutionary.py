import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from dataset import transformed_dataset
from model import NeuralNetwork
from torch.utils.data import random_split
import math
import matplotlib.pyplot as plt
from dataset import transformed_dataset

model = NeuralNetwork()
parameters = [[name, p] for name, p in model.named_parameters()]

names = [name for name, p in parameters]
def create_random_tensor(*args):
    array = np.random.rand(*args) - 0.5
    return array

def create_param_sets(gen_size):
    sets = []
    for _ in range(gen_size):
        tmp = []
        tmp.append(create_random_tensor(64, 64))
        tmp.append(create_random_tensor(64))
        tmp.append(create_random_tensor(64, 64))
        tmp.append(create_random_tensor(64))
        tmp.append(create_random_tensor(64, 64))
        tmp.append(create_random_tensor(64))
        tmp.append(create_random_tensor(10, 64))
        tmp.append(create_random_tensor(10))
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
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            # test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # test_loss /= num_batches
    correct /= len(dataloader.dataset)
    return correct


def evaluate(params, dataloader):
    load_params_to_model(params)
    score = eval_loop(dataloader, model, nn.CrossEntropyLoss())
    return score


def evaluate_all(param_list, dataloader):
    scores = []
    for elem in param_list:
        scores.append(evaluate(elem, dataloader))
    return [x for x in scores]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    temp = 20
    e_x = np.exp(x * temp)
    return e_x / e_x.sum()


# uses random resetting
def mutate_reset(array, mutation_prob):

    mask = np.random.randint(0,1/mutation_prob,size=array.shape).astype(bool)
    # currently generates random values in the range <-0.1, 0.1>
    r = (np.random.rand(*array.shape) - 0.5) * 0.2
    array[mask] = r[mask]
    return array

# slightly modifies the values of array
def mutate_change(array, mutation_prob):
    mutation_num = 3
    mutation_prob /= mutation_num
    mask = np.random.randint(0,1/mutation_prob,size=array.shape).astype(bool)

    for _ in range(mutation_num):
        mask = np.random.randint(0,int(1/mutation_prob),size=array.shape).astype(bool)
        array[mask] += array[mask] * (np.random.rand() - 0.5) * 0.2
    
    return array

def mutate_set(params_set, mutation_probs):
    mutation_factor = 1
    new_set = []
    for i in range(len(params_set)):
        # reset_prob = (mutation_probs[i] / mutation_factor)*(21-epoch_counter)/20
        # change_prob = (mutation_probs[i] / mutation_factor)*(epoch_counter+1)/20
        # temp = mutate_reset(params_set[i], reset_prob)
        # new_set.append(mutate_change(temp, change_prob))
        temp = mutate_reset(params_set[i], mutation_probs[i] / mutation_factor)
        new_set.append(mutate_change(temp, mutation_probs[i] / mutation_factor))

    return new_set


def cross_breed(pair):
    mother, father = pair

    child1, child2 = [], []
    for mom, dad in zip(mother, father):
        split = np.random.randint(2, len(father)-2)
        child1.append(np.concatenate((dad[:split], mom[split:])))
        child2.append(np.concatenate((mom[:split], dad[split:])))
    
    return child1, child2

def choose_parent_indexes(num_of_pairs, prob_distribution):
    # num of pairs - how many pairs to create
    # index_range - what indexes to select from
    pairs = []

    for _ in range(num_of_pairs):
        selected_indexes = np.random.choice(np.arange(len(prob_distribution)), size=2, p=prob_distribution, replace=False)
        pairs.append(selected_indexes)
    
    return pairs


def create_generation(params_sets, prob_distribution, gen_size, layer_mutate_prob):
    
    pairs = choose_parent_indexes((gen_size // 2) + 1, prob_distribution)
    new_gen = []

    for pair in pairs:
        # this for loop creates kids from pairs of params
        pair = [params_sets[x] for x in pair]
        new_gen += cross_breed(pair)
    
    for i in range(gen_size):
        # this for loop is responsible for mutations in kids
        new_gen[i] = mutate_set(new_gen[i], layer_mutate_prob)

    return new_gen[:gen_size]


def create_graph(values, values2):
    plt.plot(values)
    plt.plot(values2)
    plt.savefig("plot.png")


def evolve(epochs, gen_size, dataloader):

    params_sets = create_param_sets(gen_size) # begin with random arrays of parameters
    elitism_factor = 0.5
    average_accuracy = []
    max_accuracy = []
    for _ in range(epochs):
        
        # the following lines create tables to plot for later debugging 
        accuracies = np.array(evaluate_all(params_sets, dataloader))
        average_accuracy.append(sum(accuracies)/gen_size)
        max_accuracy.append(max(accuracies))
        
        prob_distribution = softmax(accuracies)
        # selected_index = np.random.choice(np.arange(gen_size), size=gen_size//2, p=prob_distribution, replace=False)
        # selected_params = [params_sets[x] for x in selected_index]

        temp = sorted(list(zip(prob_distribution, params_sets)), reverse=True, key=lambda x:x[0])[:int(gen_size*elitism_factor)]
        temp = list(list(zip(*temp))[1])
        params_sets = temp + create_generation(params_sets, prob_distribution, math.ceil(gen_size*(1-elitism_factor)), layer_mutate_prob=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
        
    create_graph(average_accuracy, max_accuracy)
    temp = list(zip(evaluate_all(params_sets, dataloader), params_sets))
    temp = sorted(temp, key=lambda x: x[0])
    print(temp[0][0], temp[-1][0])
    return temp[-1][1]
    
if __name__ == "__main__":
    datasets = random_split(transformed_dataset, (898, 899))
    train_dataloader = DataLoader(datasets[0], batch_size=40, shuffle=True)
    test_dataloader = DataLoader(datasets[1], batch_size=40, shuffle=True)
    params = evolve(200, 100, train_dataloader)
    print("Evolved accuracy: " + str(evaluate(params, test_dataloader)))
    