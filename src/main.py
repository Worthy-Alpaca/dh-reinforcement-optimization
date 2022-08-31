from collections import namedtuple
from genericpath import exists
import math
import os
import sys
from pathlib import Path
from types import FunctionType
from typing import NamedTuple
from unittest import result

from matplotlib import pyplot as plt
from model import QNetModel
from helper import QFunction
from validate import Validate
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
from datetime import datetime
from misc.deploy import DeployModel
import pandas as pd

"""PACKAGE_PARENT = "../"
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))"""
from helper import Memory, UtilFunctions, Cartsetup, Coating

from misc.dataloader import DataLoader, DataBaseLoader, KappaLoader

# from src.helper.dataloader import DataLoader
# from helper.deploy import DeployModel


class RunModel:
    def __init__(
        self,
        dbpath: str,
        numSamples: int = 10,
        tuning: bool = False,
        allowDuplicates: bool = False,
    ) -> None:
        """Initiate the reinforcement learning model and training or prediction capabilities.

        Args:
            numSamples (int, optional): The batch size for training iterations. Defaults to 10.
            tuning (bool, optional): Whether the model is in training mode. Dictates if saved models will be deleted. Defaults to False.
        """
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        try:
            if exists("./models"):
                self.folder_name = "./models"
            else:
                raise KeyError
        except:
            self.folder_name = self.resource_path("bin/assets/models")

        if tuning:
            if os.path.exists(self.folder_name):
                shutil.rmtree(self.folder_name)

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
        self.allowDuplicates = allowDuplicates
        if exists(os.getcwd() + os.path.normpath("/FINAL MODEL")):
            modelPath = Path(os.getcwd() + os.path.normpath("/FINAL MODEL"))
        else:
            modelPath = Path(self.resource_path("bin/assets/final model"))
        self.model = DeployModel(modelPath)
        self.engine = create_engine(f"sqlite:///{dbpath}")
        self.training = True
        dbData = self.engine.execute("SELECT * FROM 'products'").fetchall()
        prodData = []
        for i in dbData:
            prodData.append(i[0])

        # prodData = random.sample(prodData, self.numSamples)
        for i in tqdm(prodData):
            product = i
            overallLen = 0
            overallTime = 0
            allComponents = []
            for m in ["m10", "m20"]:

                dataloader = DataBaseLoader(self.engine, i, prodData)
                data, components, offsets, score = dataloader()
                rowOffsets = offsets / 2
                rowOffsets = offsets / rowOffsets
                if len(data) == 0:
                    Ymax = 0
                    Xmax = 0
                else:
                    Ymax = data["Y"].max() * rowOffsets
                    Xmax = data["X"].max() * rowOffsets
                predArray = np.array(
                    [len(data) * rowOffsets / 2, 0 if m == "m10" else 1, 0, 0]
                )

                overallTime += self.model.predict(predArray).item()
            overallTime = overallTime / 2
            overallTime += Coating(Ymax * rowOffsets)
            # overallTime += Cartsetup(components)
            overallLen += len(data) * rowOffsets

            allComponents.append(components)

            self.products[product] = {
                "len": overallLen,
                "time": overallTime,
                "score": score,
                "comps": list(
                    dict.fromkeys(list(itertools.chain.from_iterable(allComponents)))
                ),
            }
        print("Data generation complete")
        if numSamples == -1:
            self.numSamples = len(self.products.keys())

    def init_model(
        self,
        fname: str = None,
        EMBEDDING_DIMENSIONS: int = 10,
        EMBEDDING_ITERATIONS_T: int = 2,
        INIT_LR: float = 3e-3,
        OPTIMIZER: FunctionType = torch.optim.Adam,
        LR_DECAY_RATE: float = 1.0 - 2e-5,
    ):
        """Initiate the model state loading. If no `fname` is given, a new model will be initialized.

        Args:
            fname (str, optional): Path to a saved state_dict. Defaults to None.
            EMBEDDING_DIMENSIONS (int, optional): Embedding dimensions in the model. Defaults to 10.
            EMBEDDING_ITERATIONS_T (int, optional): Embedding iterations in the model. Defaults to 2.
            INIT_LR (float, optional): Initial learning rate. Changes over time. Defaults to 3e-3.
            OPTIMIZER (FunctionType, optional): The Optimizer to use. Defaults to torch.optim.Adam.
            LR_DECAY_RATE (float, optional): The decay rate of the initial learning rate. Defaults to 1.0-2e-5.

        Returns:
            tuple: Returns essential components of the training and prediction process: Q_func, Q_net, optimizer, lr_scheduler
        """
        self.embedding_dimensions = EMBEDDING_DIMENSIONS
        self.embedding_iterations_t = EMBEDDING_ITERATIONS_T
        Q_net = QNetModel(device=self.device, emb_dim=EMBEDDING_DIMENSIONS, T=EMBEDDING_ITERATIONS_T)
        optimizer = OPTIMIZER(Q_net.parameters(), lr=INIT_LR)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=LR_DECAY_RATE
        )
        timestamp = datetime.now().strftime("%m-%d-%Y_%H_%M_%S")
        self.run_name = (
            f"{OPTIMIZER.__class__.__name__}_{EMBEDDING_DIMENSIONS}@{timestamp}"
        )
        self.writer = SummaryWriter(
            os.getcwd() + os.path.normpath(f"/tensorboard/{self.run_name}")
        )

        coords, W_np, _ = self.getData()

        W = torch.tensor(
            W_np, dtype=torch.float32, requires_grad=False, device=self.device
        )
        solution = [random.randint(0, coords.shape[0] - 1)]
        current_state = self.State(
            partial_solution=solution, W=W, coords=coords[:, :3].astype(np.float32)
        )
        state_tsr = self.state2tens(current_state)

        # summary(Q_net, (state_tsr.to('cpu').unsqueeze(0).shape, W.to('cpu').unsqueeze(0).shape))
        # with torch.no_grad():
        #     self.writer.add_graph(Q_net, (state_tsr.unsqueeze(0), W.unsqueeze(0)))

        if fname is not None:
            checkpoint = torch.load(fname)
            Q_net.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        Q_func = QFunction(Q_net, optimizer, lr_scheduler, device=self.device)
        return Q_func, Q_net, optimizer, lr_scheduler

    def checkpoint_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim,
        lr_scheduler: torch.optim,
        loss: float,
        episode: int,
        avg_length: float,
    ):
        """Method to checkpoint the model.

        Args:
            model (torch.nn.Module): The current model.
            optimizer (torch.optim): The current optimizer.
            lr_scheduler (torch.optim): the current learning rate scheduler.
            loss (float): The current loss.
            episode (int): The current episode.
            avg_length (float): The current average length.
        """
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        fname = os.path.join(self.folder_name, "ep_{}".format(episode))
        fname += "_emb_{}".format(self.embedding_dimensions)
        fname += "_it_{}".format(self.embedding_iterations_t)
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

    def plot_solution(self, coords: np.ndarray, solution: list, validate=False):
        """Method to plot the given coordinates according to the give solution.

        Args:
            coords (np.ndarray): The current coordinate set.
            solution (list): The calculated solution.
        """

        labels = coords[:, 3:4]
        labels = labels[:, 0].tolist()

        solutionList = []
        for x in solution:
            solutionList.append(coords[x][3:4][0])

        if not validate:
            solutionList = self.calcGroups(solutionList)
        SETUPMINUTES = 10
        # groupTimings = len(solutionList) * SETUPMINUTES * 60
        groupTimings = 0
        textstr = f"{len(solutionList)} Groups\n"
        testArr = []
        for x in solutionList:
            textstr += f"{x}\n"
            runningArr = []
            if not validate:
                for i in x:
                    runningArr.append(labels.index(i))
                testArr.append(runningArr)

        coords = coords[:, :3].astype(np.float32)
        fig = plt.figure()
        plot = fig.add_subplot(121, projection="3d")
        ax = fig.add_subplot(122)
        ax.axis("off")
        plot.scatter(coords[:, 0], coords[:, 1], coords[:, 2])

        # for i, label in enumerate(labels):
        #     x = coords[:, 0][i]
        #     y = coords[:, 1][i]
        #     z = coords[:, 2][i]
        #     plot.annotate(label, (x, y, z))

        n = len(coords)

        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            wrap=True,
        )
        for idx in range(n - 1):
            i, next_i = solution[idx], solution[idx + 1]
            plot.plot(
                [coords[i, 0], coords[next_i, 0]],
                [coords[i, 1], coords[next_i, 1]],
                [coords[i, 2], coords[next_i, 2]],
                "k",
                lw=2,
                alpha=0.8,
            )

        i, next_i = solution[-1], solution[0]
        plot.plot(
            [coords[i, 0], coords[next_i, 0]],
            [coords[i, 1], coords[next_i, 1]],
            [coords[i, 2], coords[next_i, 2]],
            "k",
            lw=2,
            alpha=0.8,
        )
        plot.set(
            xlabel="Number of placements",
            ylabel="Number of Components",
            zlabel="Cumulative Component Score",
        )
        # plot.xlabel("Number of placements")
        # plot.ylabel("Number of Components")
        plot.plot(
            coords[solution[0], 0],
            coords[solution[0], 1],
            coords[solution[0], 2],
            "x",
            markersize=10,
        )
        return groupTimings

    def state2tens(self, state: NamedTuple) -> torch.tensor:
        """Method to convert a given state into PyTorch tensor.

        Args:
            state (NamedTuple): The current state.

        Returns:
            torch.tensor: The created tensor.
        """
        solution = set(state.partial_solution)
        sol_last_node = (
            state.partial_solution[-1] if len(state.partial_solution) > 0 else -1
        )
        sol_first_node = (
            state.partial_solution[0] if len(state.partial_solution) > 0 else -1
        )
        coords = state.coords[:, :3].astype(np.float32)
        nr_nodes = coords.shape[0]

        xv = [
            [
                (1 if i in solution else 0),
                (1 if i == sol_first_node else 0),
                (1 if i == sol_last_node else 0),
                coords[i, 0],
                coords[i, 1],
                coords[i, 2],
            ]
            for i in range(nr_nodes)
        ]

        return torch.tensor(
            xv, dtype=torch.float32, requires_grad=False, device=self.device
        )

    def getRandomSample(self, size: int, notAllowed: list = False):
        """Method to retrieve a random sample based on the given size.

        Args:
            size (int): The size needed.

        Returns:
            list: The generated sample.
        """
        allowed = list(self.products.keys()).copy()
        if notAllowed:
            _ = [allowed.remove(item) for item in notAllowed if item in allowed]
        listkeys = np.random.choice(
            len(allowed), size=size, replace=self.allowDuplicates
        )

        return [list(allowed)[x] for x in listkeys]

    def get_duplicates(self, iterable: list):
        proofingDict = {}
        for el in iterable:
            try:
                proofingDict[el] += 1
            except:
                proofingDict[el] = 1
        duplicates = []
        occurances = []
        for key in proofingDict.keys():
            for i in range(proofingDict[key] - 1):
                duplicates.append(key)
                occurances.append(iterable.count(key))
        return duplicates, occurances

    def getData(
        self, key: bool = False, samples: list = False, sampleReqsList: list = False
    ):
        """Method to get a data sample.

        Args:
            key (bool, optional): Returns the data for this key if given. Defaults to False.
            samples (list, optional): Generates the data from the given list of samples. Defaults to False.
        """

        def compare(p1, p2):
            l2 = list(set(p1) & set(p2))
            l1 = len(p1)
            return l2

        if samples:
            x = compare(self.products, samples)
            if len(x) != len(samples):
                print("something")
            sampleSize = samples
        else:
            sampleSize = self.getRandomSample(self.numSamples)

        if sampleReqsList:
            sampleReqs = sampleReqsList
        else:
            sampleReqs = np.random.randint(1, 70, size=(len(sampleSize))).tolist()

        currentDuplicates, occurances = self.get_duplicates(sampleSize)
        for x in currentDuplicates:
            lettersIndexes = [i for i in range(len(sampleSize)) if sampleSize[i] == x]
            runningReqs = 0
            for i in lettersIndexes:
                runningReqs += sampleReqs[i]
            lettersIndexes.reverse()
            for i in lettersIndexes:
                del sampleReqs[i]
                del sampleSize[i]
            lettersIndexes.reverse()
            sampleReqs.insert(lettersIndexes[0], runningReqs)
            sampleSize.insert(lettersIndexes[0], x)
        # sampleSize = list(set(sampleSize))
        def createCoords(sampleSize):

            globalList = {}
            currentDict = []
            for i in sampleSize:
                currentList = sampleSize.copy()
                currentList.remove(i)
                req = sampleReqs[sampleSize.index(i)]
                simTime = (
                    self.products[i]["time"] if self.products[i]["time"] != 0 else 0
                )
                if simTime > 1000:
                    simTime = simTime / 100
                currentDict.append(
                    [
                        math.log(float(self.products[i]["len"])) / 10
                        if float(self.products[i]["len"]) != 0
                        else 0,
                        math.log(simTime) / 10,
                        math.log(self.products[i]["score"]) / 10
                        if self.products[i]["score"] != 0
                        else 0,
                        i,
                        self.products[i]["comps"],
                        float(len(self.products[i]["comps"])),
                        req,
                    ]
                )

                globalList[i] = currentDict
            return globalList

        globalList = createCoords(sampleSize)
        npArray = []

        if len(globalList) < self.numSamples and self.training:
            supplement = self.getRandomSample(
                self.numSamples - len(globalList), sampleSize
            )
            supplement = supplement + sampleSize
            globalList = createCoords(supplement)
            # globalList = dict(globalList, **supplement)

        product, coords = random.choice(list(globalList.items()))

        for i in coords:
            npArray.append(i)

        coords = np.asarray(coords, dtype=object)
        # test = coords[:, :3].astype(np.float32)
        W_np = distance_matrix(
            coords[:, :3].astype(np.float32), coords[:, :3].astype(np.float32)
        )
        # test = self.distance_matrix(coords)
        # if W_np.shape != (self.numSamples, self.numSamples):
        #     print()
        return coords, W_np, product

    def distance_matrix(self, coords: np.ndarray):
        """Create a custom distance matrix based on provided coordinate set.

        Args:
            coords (np.ndarray): The given coordinate set.

        Returns:
            np.ndarray: The created matrix.
        """
        nextItems = coords.copy()
        global_matrix = []
        for i in range(len(coords)):
            running_matrix = []
            numPlacements = coords[i][0]
            numComps = coords[i][1]
            product = coords[i][2]
            components = coords[i][3]
            assemblyTime = coords[i][5]
            productRequirement = coords[i][6]
            setupTime = Cartsetup(components)
            assemblyTime = assemblyTime * productRequirement
            for j in range(len(coords)):
                numPlacementsNext = coords[j][0]
                numCompsNext = coords[j][1]
                productNext = coords[j][2]
                componentsNext = coords[j][3]
                assemblyTimeNext = coords[j][5]
                productRequirementNext = coords[j][6]
                setupTimeNext = Cartsetup(componentsNext)
                assemblyTimeNext = assemblyTimeNext * productRequirementNext

                overlap = list(set(components) & set(componentsNext))
                overlapTime = Cartsetup(overlap)
                if product == productNext:
                    timeDifference = 0
                else:
                    timeDifference = (setupTimeNext - overlapTime + setupTime) + (
                        assemblyTimeNext - assemblyTime
                    )
                running_matrix.append(timeDifference)
                # running_matrix.append(math.sqrt(timeDifference**2))

            global_matrix.append(running_matrix)

        return np.asarray(global_matrix)

    def get_graph_mat(self, n=10, size=1):
        """Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
        """
        coords = size * np.random.uniform(size=(n, 3))
        dist_mat = distance_matrix(coords, coords)
        return coords, dist_mat

    def fit(
        self,
        Q_func: FunctionType,
        Q_net: nn.Module,
        optimizer: FunctionType,
        lr_scheduler: FunctionType,
        NR_EPISODES: int,
        MIN_EPSILON: float,
        EPSILON_DECAY_RATE: float,
        N_STEP_QL: int,
        BATCH_SIZE: int,
        GAMMA: float,
    ):
        """Train the current model.

        Args:
            Q_func (FunctionType): The current Q_function
            Q_net (nn.Module): The current model.
            optimizer (FunctionType): The current optimizer function.
            lr_scheduler (FunctionType): The current learning rate scheduler.
            NR_EPISODES (int): The number of training iterations.
            MIN_EPSILON (float): The minimum epsilon value.
            EPSILON_DECAY_RATE (float): The current epsilon decay rate.
            N_STEP_QL (int): Placeholder
            BATCH_SIZE (int): The current batch size.
            GAMMA (float): The current gamma value.
        """
        found_solutions = dict()  # episode --> (coords, W, solution)
        self.training = True
        self.losses = []
        self.lrs = []
        self.path_lengths = []
        current_min_med_length = float("inf")
        for episode in tqdm(range(NR_EPISODES)):
            # sample a new random graph
            coords, W_np, product = self.getData()
            self.helper = UtilFunctions(coords)
            # coords, W_np = self.get_graph_mat(self.numSamples)
            W = torch.tensor(
                W_np, dtype=torch.float32, requires_grad=False, device=self.device
            )

            # current partial solution - a list of node index
            solution = [random.randint(0, coords.shape[0] - 1)]

            # current state (tuple and tensor)
            current_state = self.State(
                partial_solution=solution, W=W, coords=coords[:, :3].astype(np.float32)
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
                rwNext, next_solution = self.helper.total_distance(next_solution, W)
                rwNow, solution = self.helper.total_distance(solution, W)
                reward = -(rwNext - rwNow)

                next_state = self.State(
                    partial_solution=next_solution,
                    W=W,
                    coords=coords[:, :3].astype(np.float32),
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
                    self.losses.append(loss)
                    self.lrs.append(Q_func.optimizer.param_groups[0]["lr"])
                    self.writer.add_scalar("Loss", loss, t)
                    self.writer.add_scalar(
                        "LearningRate", Q_func.optimizer.param_groups[0]["lr"], t
                    )

                    med_length = np.median(self.path_lengths[-100:])
                    if med_length < current_min_med_length:
                        current_min_med_length = med_length
                        self.checkpoint_model(
                            Q_net, optimizer, lr_scheduler, loss, episode, med_length
                        )

            length, solution = self.helper.total_distance(solution, W)
            self.path_lengths.append(length)
            self.writer.add_scalar("Pathlength", length, episode)
            self.__createTensorboardLogs(Q_net, episode)

            if episode % 10 == 0:
                print(
                    "Ep %d. Loss = %.3f / median length = %.3f / last = %.4f / epsilon = %.4f / lr = %.4f"
                    % (
                        episode,
                        (-1 if loss is None else loss),
                        np.median(self.path_lengths[-50:]),
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
        self.plotMetrics()

    def __createTensorboardLogs(self, model: nn.Module, epoch):
        for name, module in model.named_children():
            try:
                self.writer.add_histogram(f"{name}.bias", module.bias, epoch)
                self.writer.add_histogram(f"{name}.weight", module.weight, epoch)
                self.writer.add_histogram(
                    f"{name}.weight.grad", module.weight.grad, epoch
                )
            except Exception as e:
                continue

    def plotMetrics(self):
        """Method to plot and visualize saved metrics."""

        def _moving_avg(x, N=10):
            return np.convolve(np.array(x), np.ones((N,)) / N, mode="valid")

        plt.figure(figsize=(8, 5))
        plt.semilogy(_moving_avg(self.losses, 100))
        plt.ylabel("loss")
        plt.xlabel("training iteration")

        plt.figure(figsize=(8, 5))
        plt.semilogy(_moving_avg(self.lrs, 100))
        plt.ylabel("learning rate")
        plt.xlabel("training iteration")

        plt.figure(figsize=(8, 5))
        plt.plot(_moving_avg(self.path_lengths, 100))
        plt.ylabel("average length")
        plt.xlabel("episode")
        plt.show()

    def getBestOder(
        self,
        samples: list,
        plot: bool = False,
        numCarts: int = 6,
        sampleReqs: list = False,
    ):
        """Method to predict the best order of the given sample set

        Args:
            samples (list): The current sample list
            plot (bool, optional): If the prediction should be plotted. Defaults to False.
            numCarts (int, optional): How many cartbays can be used alltogether. Defaults to 6.
            validate (bool, optional): If the validation dataset should be used. Defaults to False.
            sampleReqs (list, optional): The required production values for the validation set. Defaults to False.

        Returns:
            tuple: The best value and the best solution.
        """
        self.numCarts = numCarts
        self.training = False
        all_lengths_fnames = [
            f for f in os.listdir(self.folder_name) if f.endswith(".tar")
        ]
        shortest_fname = sorted(
            all_lengths_fnames, key=lambda s: float(s.split(".tar")[0].split("_")[-1])
        )[0]
        fname = shortest_fname.split("_")
        emb = int(fname[fname.index("emb") + 1])
        it = int(fname[fname.index("it") + 1])

        print(
            "shortest avg length found: {} with {} dimensions and {} iterations ".format(
                shortest_fname.split(".tar")[0].split("_")[-1], emb, it
            )
        )

        Q_func, Q_net, optimizer, lr_scheduler = self.init_model(
            os.path.join(self.folder_name, shortest_fname),
            EMBEDDING_DIMENSIONS=emb,
            EMBEDDING_ITERATIONS_T=it,
        )
        best_solution = {}
        best_value = float("inf")

        # for i in samples:
        coords, W_np, _ = self.getData(samples=samples, sampleReqsList=sampleReqs)
        self.helper = UtilFunctions(coords)
        # plot_graph(coords, 1)
        # plt.show()
        # coords, W_np = get_graph_mat(n=NR_NODES)
        W = torch.tensor(
            W_np, dtype=torch.float32, requires_grad=False, device=self.device
        )

        solution = [random.randint(0, coords.shape[0] - 1)]
        current_state = self.State(
            partial_solution=solution, W=W, coords=coords[:, :3].astype(np.float32)
        )
        current_state_tsr = self.state2tens(current_state)

        while not self.helper.is_state_final(current_state):
            next_node, est_reward = Q_func.get_best_action(
                current_state_tsr, current_state
            )

            solution = solution + [next_node]
            current_state = self.State(
                partial_solution=solution,
                W=W,
                coords=coords[:, :3].astype(np.float32),
            )
            current_state_tsr = self.state2tens(current_state)

        if self.helper.total_distance(solution, W)[0] < best_value:
            best_value, solution = self.helper.total_distance(solution, W)
            best_solution = {
                "W": W,
                "solution": solution,
                "coords": coords,
                "numCarts": self.numCarts,
                "products": self.products,
            }

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

    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    random.seed(1000)
    np.random.seed(1000)
    torch.manual_seed(1000)
    START_TIME = time.perf_counter()
    EMBEDDING_DIMENSIONS = 10
    EMBEDDING_ITERATIONS_T = 2
    runmodel = RunModel(
        dbpath=r"C:\Users\stephan.schumacher\Documents\repos\dh-reinforcement-optimization\products.db",
        numSamples=13,
        tuning=False,
        allowDuplicates=False,
    )

    all_lengths_fnames = [
        f for f in os.listdir(runmodel.folder_name) if f.endswith(".tar")
    ]
    shortest_fname = sorted(
        all_lengths_fnames, key=lambda s: float(s.split(".tar")[0].split("_")[-1])
    )[0]
    fname = shortest_fname.split("_")
    emb = int(fname[fname.index("emb") + 1])
    it = int(fname[fname.index("it") + 1])

    Q_Function, QNet, Adam, ExponentialLR = runmodel.init_model(
        fname=os.path.join(runmodel.folder_name, shortest_fname),
        EMBEDDING_DIMENSIONS=emb,
        EMBEDDING_ITERATIONS_T=it,
        OPTIMIZER=torch.optim.Adam,
    )
    runmodel.fit(
        Q_func=Q_Function,
        Q_net=QNet,
        optimizer=Adam,
        lr_scheduler=ExponentialLR,
        NR_EPISODES=3001,
        MIN_EPSILON=0.7,
        EPSILON_DECAY_RATE=6e-4,
        N_STEP_QL=4,
        BATCH_SIZE=32,
        GAMMA=0.7,
    )

    END_TIME = time.perf_counter() - START_TIME
    print(f"This run took {END_TIME} seconds | {END_TIME / 60} Minutes")

    path = Path(os.getcwd() + os.path.normpath("/2days.xlsx"))

    loader = KappaLoader(
        path,
        dbpath=r"C:\Users\stephan.schumacher\Documents\repos\dh-reinforcement-optimization\products.db",
    )

    samples, sampleReqs = loader()
    runmodel = RunModel(
        dbpath=r"C:\Users\stephan.schumacher\Documents\repos\dh-reinforcement-optimization\products.db",
        numSamples=len(samples),
        tuning=False,
        allowDuplicates=True,
    )

    best_value, best_solution = runmodel.getBestOder(
        sampleReqs=sampleReqs, samples=samples, plot=True, numCarts=3
    )
    validate = Validate(
        best_value,
        best_solution,
        dbpath=r"C:\Users\stephan.schumacher\Documents\repos\dh-reinforcement-optimization\products.db",
        calcGroups=True,
        overlapThreshhold=0.5,
    )
    validate.plotSoltions()
