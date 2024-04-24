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
import random

# parameters = [[name, p] for name, p in model.named_parameters()]

# names = [name for name, p in parameters]
class EvolutionaryAlgorithm:

    model = NeuralNetwork()
    gen_size = 10
    dataloader = None

    mutation_num = 3
    reset_factor = 1
    change_factor = 5
    pairs_at_once = 3

    mutation_chance = 6
    mutate_factor = 1

    layer_mutate_probs = [0.001, 0.001, 0.002, 0.002, 0.003, 0.003, 0.004, 0.004]
    elitism_factor = 0.2

    def _create_random_tensor(self, *args):
        array = (np.random.rand(*args) - 0.5)*0.6
        return array

    def _create_param_sets(self):
        sets = []
        for _ in range(self.gen_size):
            tmp = []
            tmp.append(self._create_random_tensor(64, 64))
            tmp.append(self._create_random_tensor(64))
            tmp.append(self._create_random_tensor(64, 64))
            tmp.append(self._create_random_tensor(64))
            tmp.append(self._create_random_tensor(64, 64))
            tmp.append(self._create_random_tensor(64))
            tmp.append(self._create_random_tensor(10, 64))
            tmp.append(self._create_random_tensor(10))
            sets.append(tmp)

        return sets


    def _load_params_to_model(self, params):
        pretrained_dict = self.model.state_dict()
        counter = 0

        for key in pretrained_dict.keys():
            pretrained_dict[key] = torch.from_numpy(params[counter])
            counter += 1

        self.model.load_state_dict(pretrained_dict)


    def _eval_loop(self, dataloader=None):
        if not dataloader:
            dataloader = self.dataloader

        self.model.eval()
        # num_batches = len(dataloader)
        correct = 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self.model(X)
                # test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # test_loss /= num_batches
        correct /= len(dataloader.dataset)
        return correct


    def evaluate(self, params, dataloader=None):
        if not dataloader:
            dataloader = self.dataloader
        self._load_params_to_model(params)
        score = self._eval_loop()
        return score


    def _evaluate_all(self, param_list):
        scores = []
        for elem in param_list:
            scores.append(self.evaluate(elem))
        return [x for x in scores]

    @staticmethod
    def _softmax(x):
        """Compute softmax values for each sets of scores in x."""
        temp = 5
        e_x = np.exp(x * temp)
        return e_x / e_x.sum()


    # uses random resetting
    def _mutate_reset(self, array, mutation_prob):

        mask = np.random.randint(0,1/mutation_prob,size=array.shape).astype(bool)
        # currently generates random values in the range <-0.25, 0.25>
        r = (np.random.rand(*array.shape) - 0.5) * 0.5
        array[mask] = r[mask]
        return array

    # slightly modifies the values of array
    def _mutate_change(self, array, mutation_prob):

        mutation_prob /= self.mutation_num

        for _ in range(self.mutation_num):
            mask = np.random.randint(0,int(1/mutation_prob),size=array.shape).astype(bool)
            array[mask] += array[mask] * (np.random.rand() - 0.5) * 0.05

        return array

    def _mutate_set(self, params_set):

        new_set = []
        for i in range(len(params_set)):
            # reset_prob = (mutation_probs[i] / mutation_factor)*(21-epoch_counter)/20
            # change_prob = (mutation_probs[i] / mutation_factor)*(epoch_counter+1)/20
            # temp = mutate_reset(params_set[i], reset_prob)
            # new_set.append(mutate_change(temp, change_prob))
            temp = self._mutate_reset(params_set[i], self.layer_mutate_probs[i] * self.reset_factor * self.mutate_factor)
            new_set.append(self._mutate_change(temp, self.layer_mutate_probs[i] * self.change_factor * self.mutate_factor))

        return new_set


    def _cross_breed(self, pair):
        mother, father = pair

        child1, child2 = [], []
        for mom, dad in zip(mother, father):
            split = np.random.randint(2, len(father)-2)
            child1.append(np.concatenate((dad[:split], mom[split:])))
            child2.append(np.concatenate((mom[:split], dad[split:])))

        return child1, child2


    def random_avg(self, arr1, arr2): # returns an array with each value randomly chosen between arr1, arr2 values
        avg_arr = []
        for el1, el2 in zip(arr1, arr2):
            if el1 < el2:
                avg_arr.append(random.uniform(el1, el2))
            else:
                avg_arr.append(random.uniform(el2, el1))
        return np.array(avg_arr)

    def _average_breed(self, pair):
        mother, father = pair
        child1, child2 = [], []
        i=0
        for mom, dad in zip(mother, father): # both mom and dad are a layer
            if i%2 ==0: # some
                to_child1 = []
                to_child2 = []
                for one, two in zip (mom,dad):
                    to_child1.append(self.random_avg(one,two))
                    to_child2.append(self.random_avg(one,two))
                child1.append(np.array(to_child1))
                child2.append(np.array(to_child2))
            else:
                child1.append(self.random_avg(mom,dad))
                child2.append(self.random_avg(mom,dad))

            i += 1
        return child1, child2


    def _choose_parent_indexes(self, num_of_pairs, prob_distribution):
        # num of pairs - how many pairs to create
        # index_range - what indexes to select from
        pairs = []

        for _ in range(num_of_pairs // self.pairs_at_once + 1):
            selected_indexes = np.random.choice(np.arange(len(prob_distribution)), size=2*self.pairs_at_once, p=prob_distribution, replace=False)
            for i in range(self.pairs_at_once):
                pairs.append(selected_indexes[i*2: (i+1)*2])

        return pairs


    def _create_generation(self, params_sets, prob_distribution, num_created):

        pairs = self._choose_parent_indexes((num_created // 2) + 1, prob_distribution)
        new_gen = []

        for pair in pairs:
            # this for loop creates kids from pairs of params
            pair = [params_sets[x] for x in pair]
            new_gen += self._average_breed(pair)

        for i in range(num_created):
            # this for loop is responsible for mutations in kids

            # now only mutates 1 in mutation_chance children
            if np.random.randint(0, self.mutation_chance) == False:
                new_gen[i] = self._mutate_set(new_gen[i])

        return new_gen[:num_created]

    def _update_mutation_factor(self, ratio):
        # if ratio == 0:
        #     return 1
        # y = 1/(ratio**(1/4)) - 1
        # return y
        if ratio == 0: return
        if ratio <0.05: self.mutate_factor *= 1.01
        else: self.mutate_factor *= 0.99


    def _create_graph(self, avg, max_):
        plt.plot([100*x for x in avg], label="avg")
        plt.plot([100*x for x in max_], label="max")
        plt.xlabel("Generation")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("plot.png")


    def evolve(self, epochs, gen_size, dataloader):

        self.gen_size = gen_size
        self.dataloader = dataloader

        params_sets = self._create_param_sets() # begin with random arrays of parameters

        ratio_of_acc = 0.25 # for this the function is 1 at the start
        average_accuracy = []
        max_accuracy = []
        for _ in range(epochs):

            # the following lines create tables to plot for later debugging
            accuracies = np.array(self._evaluate_all(params_sets))
            average_accuracy.append(sum(accuracies)/self.gen_size)
            max_accuracy.append(max(accuracies))
            last_x_avg = sum(average_accuracy[-30:])/len(average_accuracy[-30:])
            all_avg= sum(average_accuracy)/len(average_accuracy)

            # used to increase mutation if algorithm has stagnated
            self._update_mutation_factor(ratio_of_acc)

            # near 0 if stable - need more mutation
            # higher - unstable - less mutation
            ratio_of_acc = abs(1 - last_x_avg/all_avg)
            print(self.mutate_factor)
            print(ratio_of_acc)
            print("\n")

            # print(mutation_factor(ratio_of_acc))
            # print(layer_mutate_prob[-1])

            prob_distribution = self._softmax(accuracies)
            # selected_index = np.random.choice(np.arange(gen_size), size=gen_size//2, p=prob_distribution, replace=False)
            # selected_params = [params_sets[x] for x in selected_index]

            temp = sorted(list(zip(prob_distribution, params_sets)), reverse=True, key=lambda x:x[0])[:int(self.gen_size*self.elitism_factor)]
            temp = list(list(zip(*temp))[1])
            params_sets = temp + self._create_generation(params_sets, prob_distribution, num_created=math.ceil(self.gen_size*(1-self.elitism_factor)))


        self._create_graph(average_accuracy, max_accuracy)
        temp = list(zip(self._evaluate_all(params_sets), params_sets))
        temp = sorted(temp, key=lambda x: x[0])
        print(temp[0][0], temp[-1][0])
        return temp[-1][1]


if __name__ == "__main__":
    my_evo = EvolutionaryAlgorithm()
    datasets = random_split(transformed_dataset, (898, 899))
    train_dataloader = DataLoader(datasets[0], batch_size=40, shuffle=True)
    test_dataloader = DataLoader(datasets[1], batch_size=40, shuffle=True)
    params = my_evo.evolve(200, 20, train_dataloader)
    print("Evolved accuracy: " + str(my_evo.evaluate(params, test_dataloader)))
