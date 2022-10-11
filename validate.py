import math
import numpy as np
import os
import pickle
import random
import time
import torch
from itertools import permutations
from pathlib import Path
from tqdm import tqdm

from src.helper import UtilFunctions
from src.main import RunModel
from src.misc.dataset import ProductDataloader, ProductDataset

basepath = Path(os.path.expanduser(os.path.normpath("~/Documents/D+H optimizer/")))


def generateData(products):
    sampleSize = list(products.keys())
    sampleReqs = np.random.randint(1, 70, size=(len(sampleSize))).tolist()
    currentDict = []
    clist = []
    for i in sampleSize:
        currentList = sampleSize.copy()
        currentList.remove(i)
        req = sampleReqs[sampleSize.index(i)]
        simTime = products[i]["time"] if products[i]["time"] != 0 else 0
        if simTime > 1000:
            simTime = simTime / 100
        clist.append(products[i]["comps"])
        currentDict.append(
            [
                math.log(float(products[i]["len"])) / 10
                if float(products[i]["len"]) != 0
                else 0,
                math.log(simTime) / 10,
                math.log(products[i]["score"]) / 10 if products[i]["score"] != 0 else 0,
                i,
                # self.products[i]["comps"],
                # float(len(self.products[i]["comps"])),
                # req,
            ]
        )
        del req, simTime, currentList
    currentDict = np.asarray(currentDict, dtype=object)
    clist = np.asarray(clist, dtype=object)
    max_len = np.max([len(a) for a in clist])
    clist = np.asarray(
        [
            np.pad(a, (0, max_len - len(a)), "constant", constant_values=-1)
            for a in clist
        ],
        dtype=np.int32,
    )
    del max_len, sampleReqs, sampleSize
    return currentDict, clist


def loadData():
    with open(basepath / "cache.p", "rb") as file:
        products = pickle.load(file)

    return products


class ValidationTest:
    def __init__(self, components, W_np) -> None:
        self.components = components
        self.W = W_np

    def leg_cost(self, city1, city2):
        """returns cost of travel from city1 to city2."""
        l1 = self.components[city1]
        l2 = self.components[city2]
        a_cat_b, counts = torch.cat([l1, l2]).unique(return_counts=True)
        l2 = a_cat_b[torch.where(counts.gt(1))]

        if l2.shape[0] == 1 and l2[0].item() == -1:
            overlap = 1e-2
        else:
            overlap = l2.shape[0] / l1.shape[0]
        del l1, l2

        distance = self.W[city1, city2].item()

        return distance / overlap

    def path_cost(self, path):
        """total cost to travel the path and return to starting city."""
        begin = time.perf_counter()
        sum_path = sum(self.leg_cost(path[n - 1], path[n]) for n in range(len(path)))
        end = time.perf_counter() - begin
        return sum_path

    def brute_force(self, cities):
        """finds shortest path by exhaustively checking all paths."""
        cities = list(range(len(cities)))
        x = min(permutations(cities), key=self.path_cost)
        return self.calc_total_time(list(x)), x

    def citySwapAlg(self, cities):
        cities = list(range(len(cities)))

        def city_swap(city_1, city_2, current_solution):
            tour_choice = current_solution.copy()
            keeper = tour_choice[city_1].copy()
            tour_choice[city_1] = tour_choice[city_2].copy()
            tour_choice[city_2] = keeper

            if self.path_cost(tour_choice) < self.path_cost(current_solution):
                current_solution = tour_choice
            return current_solution

        partly_initial_solution = np.random.permutation(range(0, len(cities)))
        initial_solution = np.append(
            partly_initial_solution, [partly_initial_solution[0]]
        )

        current_solution = partly_initial_solution

        for k in range(10):
            for i in range(1, len(cities) - 1):
                for j in range(i + 1, len(cities)):
                    current_solution = city_swap(i, j, current_solution)

        return self.calc_total_time(current_solution)

    def nearest_neighbor(self, cities):
        """finds a path through the cities using a nearest neighbor heuristic."""

        unvisited = list(range(len(cities)))
        visited = [unvisited.pop()]

        while unvisited:
            city = min(unvisited, key=lambda c: self.leg_cost(visited[-1], c))
            visited.append(city)
            unvisited.remove(city)

        return self.calc_total_time(visited), visited

    def annealing(self, cities):
        T, cooling_rate, T_lower_bound, tolerance = 100, 0.5, 0.01, 1
        h = 0
        cities = list(range(len(cities)))
        current_solution = np.random.permutation(range(len(cities)))

        def generate_solution(
            current_solution,
        ):  # A new solution will be created by swapping two random cities in the current solution
            idx1, idx2 = np.random.choice(len(current_solution), 2)
            current_solution_copy = current_solution.copy()
            current_solution_copy[idx2], current_solution_copy[idx1] = (
                current_solution_copy[idx1],
                current_solution_copy[idx2],
            )
            return current_solution_copy

        while T > T_lower_bound:
            h += 1
            while True:
                potential_solution = generate_solution(current_solution)
                potential_distance = self.path_cost(potential_solution)
                current_distance = self.path_cost(current_solution)

                if potential_distance < current_distance:
                    current_solution = potential_solution

                elif np.random.random() < np.exp(
                    -(potential_distance - current_distance) / T
                ):
                    current_solution = potential_solution

                if np.abs(potential_distance - current_distance) < tolerance:
                    break

            T = T * cooling_rate
            if h % 1000 == 0:
                print(h, T)

        return self.calc_total_time(current_solution)

    def calc_total_time(self, solution: list):
        total_overlap = 0
        r1 = 0
        for step in range(len(solution) - 1):
            idx1, idx2 = solution[step], solution[step + 1]
            c1 = self.components[idx1]
            c2 = self.components[idx2]
            a_cat_b, counts = torch.cat([c1, c2]).unique(return_counts=True)
            overlapComponents = a_cat_b[torch.where(counts.gt(1))]
            r1 += Cartsetup(c1)
            total_overlap += Cartsetup(overlapComponents)
            del c1, c2
        return total_overlap

    def neuralNetwork(self, coords, runmodel, Q_func):
        solution = [random.randint(0, coords.shape[0] - 1)]
        current_state = runmodel.State(partial_solution=solution, W=W, coords=coords)
        current_state_tsr = runmodel.state2tens(current_state)

        while not helper.is_state_final(current_state):
            next_node, est_reward = Q_func.get_best_action(
                current_state_tsr, current_state
            )

            solution = solution + [next_node]
            current_state = runmodel.State(
                partial_solution=solution, W=W, coords=coords[:, :3]
            )
            current_state_tsr = runmodel.state2tens(current_state)

        return self.calc_total_time(solution)


setup = "from __main__ import brute_force, nearest_neighbor, out"
out = [None]


def Cartsetup(comps: list):
    time = 0

    complexity = 36 / len(comps)
    for i in range(len(comps)):
        time = ((60 + random.randint(0, 30)) * complexity + 9.8) + time
    return time


NUM_SAMPLES = 24
productDataset = ProductDataset(*generateData(loadData()))
productDataloader = ProductDataloader(
    productDataset,
    NUM_SAMPLES,
    shuffle=True,
    num_workers=4,
    drop_last=False,
    persistent_workers=True,
    pin_memory=True,
)


if __name__ == "__main__":

    np.random.seed(2408)
    torch.manual_seed(2408)
    random.seed(2408)
    ITERATIONS = 10

    running_data_nn = []
    running_data_NN = []
    running_time_nn = []
    running_time_NN = []
    running_data_CS = []
    running_time_CS = []
    running_data_AN = []
    running_time_AN = []

    """ Initiate RunModel"""
    runmodel = RunModel(
        dbpath=r"C:\Users\stephan.schumacher\Documents\repos\dh-reinforcement-optimization\data\SMD_Material_Stueli.txt",
        numSamples=NUM_SAMPLES,
        overwriteDevice="cpu",
        caching=True,
    )
    all_lengths_fnames = [
        f for f in os.listdir(basepath / "models") if f.endswith(".tar")
    ]
    shortest_fname = sorted(
        all_lengths_fnames, key=lambda s: float(s.split(".tar")[0].split("_")[-1])
    )[0]
    fname = shortest_fname.split("_")
    emb = int(fname[fname.index("emb") + 1])
    it = int(fname[fname.index("it") + 1])

    Q_func, Q_net, optimizer, lr_scheduler = runmodel.init_model(
        os.path.join(basepath / "models", shortest_fname),
        EMBEDDING_DIMENSIONS=emb,
        EMBEDDING_ITERATIONS_T=it,
    )
    for coords, W_np, components in tqdm(iter(productDataloader)):
        coords_list, W_np, components = coords.tolist(), W_np, components
        W = torch.tensor(
            W_np, dtype=torch.float, requires_grad=False, device=runmodel.device
        )
        helper = UtilFunctions(components)
        validation = ValidationTest(components, W_np)
        """ Nearest Neighbor """
        START_TIME = time.perf_counter()
        best_overlap = -float("inf")
        for x in range(ITERATIONS):
            total_overlap, solution_nn = validation.nearest_neighbor(coords_list)
            if total_overlap > best_overlap:
                best_overlap = total_overlap
        END_TIME = time.perf_counter() - START_TIME
        running_data_nn.append(best_overlap)
        running_time_nn.append(END_TIME)

        """ Cityswap Algorithm """
        START_TIME = time.perf_counter()
        best_overlap = -float("inf")
        for x in range(ITERATIONS):
            total_overlap = validation.citySwapAlg(coords_list)
            if total_overlap > best_overlap:
                best_overlap = total_overlap
        END_TIME = time.perf_counter() - START_TIME
        running_data_CS.append(best_overlap)
        running_time_CS.append(END_TIME)

        """ Annealing Algorithm """
        START_TIME = time.perf_counter()
        best_overlap = -float("inf")
        for x in range(ITERATIONS):
            total_overlap = validation.annealing(coords_list)
            if total_overlap > best_overlap:
                best_overlap = total_overlap
        END_TIME = time.perf_counter() - START_TIME
        running_data_AN.append(best_overlap)
        running_time_AN.append(END_TIME)

        """
        Brute Force for 24 Nodes would take 1.6214178e+21 seconds
        """

        """ Neural Network"""
        START_TIME = time.perf_counter()
        best_overlap = -float("inf")
        for x in range(ITERATIONS):
            total_overlap = validation.neuralNetwork(coords, runmodel, Q_func)
            if total_overlap > best_overlap:
                best_overlap = total_overlap
        END_TIME = time.perf_counter() - START_TIME
        running_data_NN.append(best_overlap)
        running_time_NN.append(END_TIME)

    import matplotlib.pyplot as plt

    plt.plot(running_data_NN, label="Neural Network Algorithm")
    plt.plot(running_data_nn, label="Nearest Neighbor Algorithm")
    plt.plot(running_data_CS, label="City Swap Algorithm")
    plt.plot(running_data_AN, label="Annealing Algorithm")
    plt.legend()
    plt.xlabel("Batch Nummer")
    plt.ylabel("Berechnete gesparte Zeit durch Überschneidung in Sekunden")
    plt.show()

    plt.plot(running_time_NN, label="Neural Network Algorithm")
    plt.plot(running_time_nn, label="Nearest Neighbor Algorithm")
    plt.plot(running_time_CS, label="City Swap Algorithm")
    plt.plot(running_time_AN, label="Annealing Algorithm")
    plt.legend()
    plt.xlabel("Batch Nummer")
    plt.ylabel("Benötigte Rechenzeit für 10 Iterationen in Sekunden")
    plt.show()
