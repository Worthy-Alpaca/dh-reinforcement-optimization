from collections import namedtuple
import os
import sys
from pathlib import Path

from matplotlib import pyplot as plt
from model import QNetModel
from helper import QFunction
import numpy as np
from scipy.spatial import distance_matrix
from torch.utils.tensorboard import SummaryWriter
import random
import torch
import torch.nn as nn
import itertools
import shutil
from sqlalchemy import create_engine
from torchinfo import summary
import time
from tqdm import tqdm

from misc.deploy import DeployModel

"""PACKAGE_PARENT = "../"
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))"""
from helper import Memory, UtilFunctions

from misc.dataloader import DataLoader, DataBaseLoader

# from src.helper.dataloader import DataLoader
# from helper.deploy import DeployModel


class RunModel:
    def __init__(self, numSamples: int = 10) -> None:
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.folder_name = "./models"
        # if os.path.exists(self.folder_name):
        # shutil.rmtree(self.folder_name)

        self.products = {}
        self.memory = Memory()
        self.bestModel = {}
        path = Path(os.getcwd() + os.path.normpath("/data"))
        self.Experience = namedtuple(
            "Experience",
            ("state", "state_tsr", "action", "reward", "next_state", "next_state_tsr"),
        )
        self.State = namedtuple("State", ("W", "coords", "partial_solution"))
        self.numSamples = numSamples
        self.model = DeployModel(Path(os.getcwd() + os.path.normpath("/FINAL MODEL")))
        engine = create_engine("sqlite:///products.db")
        dbData = engine.execute("SELECT * FROM 'products'").fetchall()
        prodData = []
        for i in dbData:
            prodData.append(i[1])

        # prodData = random.sample(prodData, self.numSamples)
        print("Generating Data")
        for i in tqdm(prodData):
            product = i
            overallLen = 0
            overallTime = 0
            allComponents = []
            for m in ["m10", "m20"]:

                dataloader = DataBaseLoader(engine, i, m)
                data, components, offsets = dataloader()
                if len(data) == 0:
                    Ymax = 0
                    Xmax = 0
                else:
                    Ymax = data["Y"].max() + max(offsets[1])
                    Xmax = data["X"].max() + max(offsets[0])
                predArray = np.array(
                    [len(data) * len(offsets), 0 if m == "m10" else 1, Xmax, Ymax]
                )
                overallTime += self.model.predict(predArray).item()
                overallLen += len(data) * len(offsets)

                allComponents.append(components["index"].unique())

            self.products[product] = {
                "len": overallLen,
                "time": overallTime,
                "comps": list(itertools.chain.from_iterable(allComponents)),
            }
        print("Data generation done")

    def init_model(
        self,
        fname=None,
        EMBEDDING_DIMENSIONS=10,
        EMBEDDING_ITERATIONS_T=2,
        INIT_LR=3e-3,
        OPTIMIZER=torch.optim.Adam,
        LR_DECAY_RATE=1.0 - 2e-5,
    ):
        Q_net = QNetModel(EMBEDDING_DIMENSIONS, T=EMBEDDING_ITERATIONS_T)
        optimizer = OPTIMIZER(Q_net.parameters(), lr=INIT_LR)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=LR_DECAY_RATE
        )

        # self.writer = SummaryWriter(os.getcwd() + os.path.normpath(f"/tensorboard/"))
        coords, W_np, _ = self.getData()
        W = torch.tensor(
            W_np, dtype=torch.float32, requires_grad=False, device=self.device
        )
        solution = [random.randint(0, coords.shape[0] - 1)]
        current_state = self.State(partial_solution=solution, W=W, coords=coords)
        current_state_tsr = self.state2tens(current_state)
        # summary(Q_net, coords.shape)
        # self.writer.add_graph(Q_net)
        if fname is not None:
            checkpoint = torch.load(fname)
            Q_net.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        Q_func = QFunction(Q_net, optimizer, lr_scheduler)
        return Q_func, Q_net, optimizer, lr_scheduler

    def checkpoint_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim,
        lr_scheduler: torch.optim,
        loss,
        episode,
        avg_length,
    ):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        fname = os.path.join(self.folder_name, "ep_{}".format(episode))
        fname += "_length_{}".format(avg_length)
        fname += ".tar"

        torch.save(
            {
                "episode": episode,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "loss": loss,
                "avg_length": avg_length,
            },
            fname,
        )

    def plot_solution(self, coords, mat, solution):
        plt.scatter(coords[:, 0], coords[:, 1])
        n = len(coords)

        for idx in range(n - 1):
            i, next_i = solution[idx], solution[idx + 1]
            plt.plot(
                [coords[i, 0], coords[next_i, 0]],
                [coords[i, 1], coords[next_i, 1]],
                "k",
                lw=2,
                alpha=0.8,
            )

        i, next_i = solution[-1], solution[0]
        plt.plot(
            [coords[i, 0], coords[next_i, 0]],
            [coords[i, 1], coords[next_i, 1]],
            "k",
            lw=2,
            alpha=0.8,
        )
        plt.plot(coords[solution[0], 0], coords[solution[0], 1], "x", markersize=10)

    def state2tens(self, state):
        solution = set(state.partial_solution)
        sol_last_node = (
            state.partial_solution[-1] if len(state.partial_solution) > 0 else -1
        )
        sol_first_node = (
            state.partial_solution[0] if len(state.partial_solution) > 0 else -1
        )
        coords = state.coords[:, :2].astype(np.float32)
        nr_nodes = coords.shape[0]

        xv = [
            [
                (1 if i in solution else 0),
                (1 if i == sol_first_node else 0),
                (1 if i == sol_last_node else 0),
                coords[i, 0],
                coords[i, 1],
            ]
            for i in range(nr_nodes)
        ]

        return torch.tensor(
            xv, dtype=torch.float32, requires_grad=False, device=self.device
        )

    def getData(self, key=False):
        def compare(p1, p2):
            l2 = len(list(set(p1) & set(p2)))
            l1 = len(p1)
            return l2 / l1

        """engine = create_engine("sqlite:///products.db")
        dbData = engine.execute("SELECT * FROM 'products'").fetchall()
        prodData = []
        for i in dbData:
            prodData.append(i[1])

        prodData = random.sample(prodData, self.numSamples)

        for i in prodData:
            product = i
            overallLen = 0
            overallTime = 0
            allComponents = []
            for m in ["m10", "m20"]:

                dataloader = DataBaseLoader(engine, i, m)
                data, components, offsets = dataloader()
                if len(data) == 0:
                    Ymax = 0
                    Xmax = 0
                else:
                    Ymax = data["Y"].max() + max(offsets[1])
                    Xmax = data["X"].max() + max(offsets[0])
                predArray = np.array(
                    [len(data) * len(offsets), 0 if m == "m10" else 1, Xmax, Ymax]
                )
                overallTime += self.model.predict(predArray).item()
                overallLen += len(data) * len(offsets)

                allComponents.append(components["index"].unique())

            self.products[product] = {
                "len": overallLen,
                "time": overallTime,
                "comps": list(itertools.chain.from_iterable(allComponents)),
            }"""

        globalList = {}
        sampleSize = random.sample(list(self.products.keys()), self.numSamples)
        for i in sampleSize:
            currentList = set(sampleSize.copy())
            currentList.remove(i)
            # currentDict = [[self.products[i]["len"] / 1000, 1]]
            currentDict = [
                [
                    self.products[i]["len"],
                    self.products[i]["time"],
                    i,
                    self.products[i]["comps"],
                ]
            ]
            for j in currentList:
                currentDict.append(
                    [
                        self.products[j]["len"],
                        self.products[j]["time"],
                        j,
                        self.products[j]["comps"],
                    ]
                )
            globalList[i] = currentDict
        npArray = []
        if key == False:
            product, coords = random.choice(list(globalList.items()))
        else:
            coords = globalList[key]
            product = key
        for i in coords:
            npArray.append(i)

        coords = np.asarray(npArray, dtype=object)
        # print(coords[:, :2])
        W_np = distance_matrix(
            coords[:, :2].astype(np.float32), coords[:, :2].astype(np.float32)
        )
        return coords, W_np, product

    def get_graph_mat(self, n=10, size=1):
        """Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
        """
        coords = size * np.random.uniform(size=(n, 2))
        dist_mat = distance_matrix(coords, coords)
        return coords, dist_mat

    def fit(
        self,
        Q_func,
        Q_net,
        optimizer,
        lr_scheduler,
        NR_EPISODES,
        MIN_EPSILON,
        EPSILON_DECAY_RATE,
        N_STEP_QL,
        BATCH_SIZE,
        GAMMA,
    ):
        found_solutions = dict()  # episode --> (coords, W, solution)
        losses = []
        path_lengths = []
        current_min_med_length = float("inf")
        for episode in tqdm(range(NR_EPISODES)):
            # sample a new random graph
            coords, W_np, product = self.getData()
            self.helper = UtilFunctions(coords)
            # coords, W_np = self.get_graph_mat(len(os.listdir("./data")))
            W = torch.tensor(
                W_np, dtype=torch.float32, requires_grad=False, device=self.device
            )

            # current partial solution - a list of node index
            solution = [random.randint(0, coords.shape[0] - 1)]

            # current state (tuple and tensor)
            current_state = self.State(
                partial_solution=solution, W=W, coords=coords[:, :2].astype(np.float32)
            )
            current_state_tsr = self.state2tens(current_state)

            # Keep track of some variables for insertion in replay memory:
            states = [current_state]
            states_tsrs = [
                current_state_tsr
            ]  # we also keep the state tensors here (for efficiency)
            rewards = []
            actions = []

            # current value of epsilon
            epsilon = max(MIN_EPSILON, (1 - EPSILON_DECAY_RATE) ** episode)

            nr_explores = 0
            t = -1
            while not self.helper.is_state_final(current_state):
                t += 1  # time step of this episode

                """ ################
                check helper function
                line 116
                
                """
                if epsilon >= random.random():
                    # explore
                    next_node = self.helper.get_next_neighbor_random(current_state)
                    if next_node == None:
                        continue
                    nr_explores += 1
                else:
                    # exploit
                    next_node, est_reward = Q_func.get_best_action(
                        current_state_tsr, current_state
                    )
                    if next_node == None:
                        print()
                    if episode % 50 == 0:
                        print(
                            "Ep {} | current sol: {} / next est reward: {}".format(
                                episode, solution, est_reward
                            )
                        )

                next_solution = solution + [next_node]

                if None in solution or None in next_solution:
                    print(product)
                    print("here")

                # reward observed for taking this step
                reward = -(
                    self.helper.total_distance(next_solution, W)
                    - self.helper.total_distance(solution, W)
                )

                next_state = self.State(
                    partial_solution=next_solution,
                    W=W,
                    coords=coords[:, :2].astype(np.float32),
                )
                next_state_tsr = self.state2tens(next_state)

                # store rewards and states obtained along this episode:
                states.append(next_state)
                states_tsrs.append(next_state_tsr)
                rewards.append(reward)
                actions.append(next_node)

                # store our experience in memory, using n-step Q-learning:
                if len(solution) >= N_STEP_QL:
                    self.memory.remember(
                        self.Experience(
                            state=states[-N_STEP_QL],
                            state_tsr=states_tsrs[-N_STEP_QL],
                            action=actions[-N_STEP_QL],
                            reward=sum(rewards[-N_STEP_QL:]),
                            next_state=next_state,
                            next_state_tsr=next_state_tsr,
                        )
                    )

                if self.helper.is_state_final(next_state):
                    for n in range(1, N_STEP_QL):
                        self.memory.remember(
                            self.Experience(
                                state=states[-n],
                                state_tsr=states_tsrs[-n],
                                action=actions[-n],
                                reward=sum(rewards[-n:]),
                                next_state=next_state,
                                next_state_tsr=next_state_tsr,
                            )
                        )

                # update state and current solution
                current_state = next_state
                current_state_tsr = next_state_tsr
                solution = next_solution

                # take a gradient step
                loss = None
                if len(self.memory) >= BATCH_SIZE and len(self.memory) >= 2000:
                    experiences = self.memory.sample_batch(BATCH_SIZE)

                    batch_states_tsrs = [e.state_tsr for e in experiences]
                    batch_Ws = [e.state.W for e in experiences]
                    batch_actions = [e.action for e in experiences]
                    batch_targets = []

                    for i, experience in enumerate(experiences):
                        target = experience.reward
                        if not self.helper.is_state_final(experience.next_state):
                            _, best_reward = Q_func.get_best_action(
                                experience.next_state_tsr, experience.next_state
                            )
                            target += GAMMA * best_reward
                        batch_targets.append(target)

                    # print("batch targets: {}".format(batch_targets))
                    loss = Q_func.batch_update(
                        batch_states_tsrs, batch_Ws, batch_actions, batch_targets
                    )
                    losses.append(loss)

                    """ Save model when we reach a new low average path length
                    """
                    med_length = np.median(path_lengths[-100:])
                    if med_length < current_min_med_length:
                        current_min_med_length = med_length
                        self.checkpoint_model(
                            Q_net, optimizer, lr_scheduler, loss, episode, med_length
                        )

            length = self.helper.total_distance(solution, W)
            path_lengths.append(length)

            if episode % 10 == 0:
                print(
                    "Ep %d. Loss = %.3f / median length = %.3f / last = %.4f / epsilon = %.4f / lr = %.4f"
                    % (
                        episode,
                        (-1 if loss is None else loss),
                        np.median(path_lengths[-50:]),
                        length,
                        epsilon,
                        Q_func.optimizer.param_groups[0]["lr"],
                    )
                )
                found_solutions[episode] = (
                    W.clone(),
                    coords.copy(),
                    [n for n in solution],
                )

    def getBestOder(self, plot=False):
        all_lengths_fnames = [
            f for f in os.listdir(self.folder_name) if f.endswith(".tar")
        ]
        shortest_fname = sorted(
            all_lengths_fnames, key=lambda s: float(s.split(".tar")[0].split("_")[-1])
        )[0]
        print(
            "shortest avg length found: {}".format(
                shortest_fname.split(".tar")[0].split("_")[-1]
            )
        )

        """ Load checkpoint
        """
        Q_func, Q_net, optimizer, lr_scheduler = self.init_model(
            os.path.join(self.folder_name, shortest_fname)
        )
        best_solution = {}
        best_value = float("inf")

        for key in self.products.keys():
            #
            coords, W_np, _ = self.getData(key)
            self.helper = UtilFunctions(coords)
            # plot_graph(coords, 1)
            # plt.show()
            # coords, W_np = get_graph_mat(n=NR_NODES)
            W = torch.tensor(
                W_np, dtype=torch.float32, requires_grad=False, device=self.device
            )

            solution = [random.randint(0, coords.shape[0] - 1)]
            current_state = self.State(
                partial_solution=solution, W=W, coords=coords[:, :2].astype(np.float32)
            )
            current_state_tsr = self.state2tens(current_state)

            while not self.helper.is_state_final(current_state):
                next_node, est_reward = Q_func.get_best_action(
                    current_state_tsr, current_state
                )

                solution = solution + [next_node]
                current_state = self.State(
                    partial_solution=solution, W=W, coords=coords
                )
                current_state_tsr = self.state2tens(current_state)

            if self.helper.total_distance(solution, W) < best_value:
                best_value = self.helper.total_distance(solution, W)
                best_solution = {"W": W, "solution": solution, "coords": coords}

        if plot:
            plt.figure()
            print(
                "The best value for this iteration is: ",
                self.helper.total_distance(
                    best_solution["solution"], best_solution["W"]
                ),
            )
            print(best_solution["coords"][:, :2].astype(np.float32))
            self.plot_solution(
                best_solution["coords"][:, :2].astype(np.float32),
                best_solution["W"],
                best_solution["solution"],
            )
            plt.title(
                "model / len = {}".format(
                    self.helper.total_distance(
                        best_solution["solution"], best_solution["W"]
                    )
                )
            )
            plt.figure()
            random_solution = list(range(best_solution["coords"].shape[0]))
            self.plot_solution(
                best_solution["coords"][:, :2].astype(np.float32),
                best_solution["W"],
                random_solution,
            )
            plt.title(
                "random / len = {}".format(
                    self.helper.total_distance(random_solution, best_solution["W"])
                )
            )

            for x in best_solution["solution"]:
                print(best_solution["coords"][x][:2])
            plt.show()
        return best_value, best_solution

    def plot_graph(self, coords):
        """Utility function to plot the fully connected graph"""
        n = len(coords)

        plt.scatter(coords[:, 0], coords[:, 1], s=[50 for _ in range(n)])
        for i in range(n):
            for j in range(n):
                if j < i:
                    plt.plot(
                        [coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        "b",
                        alpha=0.7,
                    )
        plt.show()


if __name__ == "__main__":
    START_TIME = time.perf_counter()
    runmodel = RunModel(numSamples=20)
    # coords, w_np, product = runmodel.getData()
    # runmodel.plot_graph(coords)
    # exit()
    Q_Function, QNet, Adam, ExponentialLR = runmodel.init_model(
        EMBEDDING_DIMENSIONS=10, EMBEDDING_ITERATIONS_T=4
    )

    runmodel.fit(Q_Function, QNet, Adam, ExponentialLR, 3001, 0.7, 6e-4, 4, 16, 0.7)
    END_TIME = time.perf_counter() - START_TIME
    print(f"This run took {END_TIME} seconds | {END_TIME / 60} Minutes")
    for i in range(5):
        runmodel.getBestOder(plot=True)
