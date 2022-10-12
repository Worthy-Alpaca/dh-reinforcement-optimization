import math
import numpy as np
import optuna
import os
import pickle
import random
import shutil
import sys
import time
import torch
import torch.nn as nn
from collections import namedtuple
from datetime import datetime
from genericpath import exists
from logging import info
from matplotlib import pyplot as plt
from pathlib import Path
from sqlalchemy import engine
from tqdm import tqdm
from types import FunctionType
from typing import Literal, NamedTuple
from scipy.spatial import distance_matrix
from torch.utils.tensorboard import SummaryWriter

try:
    from helper import Memory, UtilFunctions, Coating
    from helper import QFunction
    from misc.dataloader import DataBaseLoader, KappaLoader
    from misc.dataset import ProductDataloader, ProductDataset
    from misc.deploy import DeployModel
    from model import QNetModel
    from validate import Validate
except:
    from src.helper import Memory, UtilFunctions, Coating
    from src.helper import QFunction
    from src.misc.dataloader import DataBaseLoader, KappaLoader
    from src.misc.dataset import ProductDataloader, ProductDataset
    from src.misc.deploy import DeployModel
    from src.model import QNetModel
    from src.validate import Validate


#######################
# Credit goes to unit8co for the base code.
# Modifications where made under the included GNU GENERAL PUBLIC LICENSE
#######################


class RunModel:
    def __init__(
        self,
        dbpath: str,
        numSamples: int = 10,
        tuning: bool = False,
        allowDuplicates: bool = False,
        overwriteDevice: Literal["cpu", "cuda:0"] = False,
        caching: bool = True,
        disableProgress: bool = False,
        refEngine: engine = None,
    ) -> None:
        """Initiate the reinforcement learning model and training or prediction capabilities.

        Args:
            numSamples (int, optional): The batch size for training iterations. Defaults to 10.
            tuning (bool, optional): Whether the model is in training mode. Dictates if saved models will be deleted. Defaults to False.
        """
        if not overwriteDevice:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = overwriteDevice.lower()

        self.folder_name = Path(
            os.path.expanduser(os.path.normpath("~/Documents/D+H optimizer/models"))
        )

        self.basepath = Path(
            os.path.expanduser(os.path.normpath("~/Documents/D+H optimizer/"))
        )

        if tuning:
            if os.path.exists(self.folder_name):
                shutil.rmtree(self.folder_name)

        self.products = {}
        self.memory = Memory()
        self.bestModel = {}
        self.Experience = namedtuple(
            "Experience",
            ("state", "state_tsr", "action", "reward", "next_state", "next_state_tsr"),
        )
        self.State = namedtuple("State", ("W", "coords", "partial_solution"))
        self.numSamples = numSamples
        self.allowDuplicates = allowDuplicates

        self.training = True
        self.disableProgress = disableProgress
        cachePath = self.basepath / "cache.p"
        if (
            exists(cachePath)
            and caching
            and cachePath.stat().st_mtime > (time.time() - 86400)
            and Path(dbpath).stat().st_mtime > (time.time() - 86400)
        ):
            try:
                with open(self.basepath / "cache.p", "rb") as file:
                    self.products = pickle.load(file)
            except Exception as e:
                raise FileNotFoundError(f"Unable to find: {self.basepath}/cache.p")
            info("Found cached data.")
        else:
            info("No cached data found or cached data too old. Generating new dataset.")
            if exists(os.getcwd() + os.path.normpath("/FINAL MODEL")):
                modelPath = Path(os.getcwd() + os.path.normpath("/FINAL MODEL"))
            else:
                modelPath = Path(self.resource_path("bin/assets/final model"))
            self.model = DeployModel(modelPath)
            dataloader = DataBaseLoader(Path(dbpath), refEngine)
            for i in tqdm(dataloader.prodData, disable=disableProgress):
                product = i
                overallLen = 0
                overallTime = 0
                data, components, offsets, score = dataloader(i)
                for m in ["m10", "m20"]:
                    rowOffsets = offsets / 2
                    rowOffsets = offsets / rowOffsets
                    if data == 0:
                        Ymax = 0
                    else:
                        Ymax = np.random.randn() * rowOffsets
                    predArray = np.array(
                        [data * rowOffsets / 2, 0 if m == "m10" else 1, 0, 0]
                    )

                    overallTime += self.model.predict(predArray).item()
                overallTime = overallTime / 2
                overallTime += Coating(Ymax * rowOffsets)
                overallLen += data * rowOffsets

                comps = [dataloader.compData.index(x.strip()) for x in components]

                self.products[product] = {
                    "len": len(comps),
                    "time": overallTime,
                    "score": score,
                    "comps": comps,
                }

            info("Data generation complete")
            if caching:
                with open(self.basepath / "cache.p", "wb") as fp:
                    info(f"Saving generated data in cache")
                    pickle.dump(self.products, fp, protocol=pickle.HIGHEST_PROTOCOL)
        if numSamples == -1:
            self.numSamples = len(self.products.keys())

    def init_model(
        self,
        fname: str = None,
        EMBEDDING_DIMENSIONS: int = 10,
        EMBEDDING_ITERATIONS_T: int = 2,
        INIT_LR: float = 3e-3,
        OPTIMIZER: FunctionType = torch.optim.Adam,
        optim_args: dict = {},
        loss_func: FunctionType = torch.nn.MSELoss,
        LR_DECAY_RATE: float = 1.0 - 2e-5,
        debug: bool = True,
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
        Q_net = QNetModel(
            device=self.device,
            emb_dim=EMBEDDING_DIMENSIONS,
            emb_it=EMBEDDING_ITERATIONS_T,
        )
        Q_net.to(self.device)
        optimizer = OPTIMIZER(Q_net.parameters(), lr=INIT_LR, **optim_args)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=LR_DECAY_RATE
        )
        timestamp = datetime.now().strftime("%m-%d-%Y_%H_%M_%S")
        self.run_name = (
            f"{OPTIMIZER.__class__.__name__}_{EMBEDDING_DIMENSIONS}@{timestamp}"
        )
        if debug:
            self.writer = SummaryWriter(
                os.getcwd() + os.path.normpath(f"/tensorboard/{self.run_name}")
            )

        if fname is not None:
            checkpoint = torch.load(fname, map_location=torch.device(self.device))
            Q_net.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        Q_func = QFunction(
            Q_net, optimizer, lr_scheduler, device=self.device, loss_fn=loss_func
        )
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
        coords = state.coords
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
            xv, dtype=torch.float, requires_grad=False, device=self.device
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

        product, coords = random.choice(list(globalList.items()))

        for i in coords:
            npArray.append(i)

        coords = np.asarray(coords, dtype=object)
        W_np = distance_matrix(
            coords[:, :3].astype(np.float32), coords[:, :3].astype(np.float32)
        )
        return coords, W_np, product

    def get_graph_mat(self, n=10, size=1):
        """Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
        """
        coords = size * np.random.uniform(size=(n, 3))
        dist_mat = distance_matrix(coords, coords)
        return coords, dist_mat

    def generateData(
        self,
    ):
        sampleSize = list(self.products.keys())
        sampleReqs = np.random.randint(1, 70, size=(len(sampleSize))).tolist()
        currentDict = []
        clist = []
        for i in sampleSize:
            currentList = sampleSize.copy()
            currentList.remove(i)
            req = sampleReqs[sampleSize.index(i)]
            simTime = self.products[i]["time"] if self.products[i]["time"] != 0 else 0
            if simTime > 1000:
                simTime = simTime / 100
            clist.append(self.products[i]["comps"])
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
        trial: optuna.trial.Trial = None,
        debug: bool = False,
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
        self.training = True
        self.losses = []
        self.lrs = []
        self.path_lengths = []
        self.rewards = []
        current_min_med_length = float("inf")
        Q_net = Q_net.float()
        # BATCH_SIZE = self.numSamples
        productDataset = ProductDataset(*self.generateData())
        productDataloader = ProductDataloader(
            productDataset,
            self.numSamples,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            persistent_workers=True,
            pin_memory=True,
        )
        step = -1
        for episode in tqdm(range(NR_EPISODES), disable=self.disableProgress):
            coords, W_np, components = next(iter(productDataloader))

            coords, W, components = (
                coords.to(self.device, non_blocking=True).float(),
                torch.tensor(
                    W_np,
                    device=self.device,
                    dtype=torch.float,
                    requires_grad=False,
                ),
                components.to(self.device, non_blocking=True),
            )
            self.helper = UtilFunctions(components)

            # current partial solution - a list of node index
            solution = [random.randint(0, coords.shape[0] - 1)]

            # current state (tuple and tensor)
            current_state = self.State(partial_solution=solution, W=W, coords=coords)
            current_state_tsr = self.state2tens(current_state)

            # Keep track of some variables for insertion in replay memory:
            states = [current_state]
            states_tsrs = [
                current_state_tsr
            ]  # we also keep the state tensors here (for efficiency)
            # rewards = torch.zeros(0, device=self.device)
            rewards = []
            actions = []

            # current value of epsilon
            epsilon = max(MIN_EPSILON, (1 - EPSILON_DECAY_RATE) ** episode)

            nr_explores = 0
            t = -1
            while not self.helper.is_state_final(current_state):
                t += 1  # time step of this episode
                step += 1
                if epsilon >= random.random():
                    # explore
                    next_node = self.helper.get_next_neighbor_random(current_state)
                    nr_explores += 1
                else:
                    # exploit
                    next_node, est_reward = Q_func.get_best_action(
                        current_state_tsr, current_state
                    )
                del current_state_tsr, current_state
                next_solution = solution + [next_node]

                # reward observed for taking this step
                reward = -(
                    self.helper.total_distance(next_solution, W)[0]
                    - self.helper.total_distance(solution, W)[0]
                )
                self.rewards.append(self.helper.total_distance(next_solution, W)[0])
                next_state = self.State(
                    partial_solution=next_solution,
                    W=W,
                    coords=coords,
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
                if len(self.memory) >= BATCH_SIZE and len(self.memory) >= 200:
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
                        del target

                    loss = Q_func.batch_update(
                        batch_states_tsrs, batch_Ws, batch_actions, batch_targets
                    )
                    self.losses.append(loss)
                    self.lrs.append(Q_func.optimizer.param_groups[0]["lr"])
                    self.writer.add_scalar("Loss", loss, t)
                    self.writer.add_scalar(
                        "LearningRate", Q_func.optimizer.param_groups[0]["lr"], t
                    )

            med_length = np.median(self.losses[-100:])
            if med_length < current_min_med_length:
                current_min_med_length = med_length
                self.checkpoint_model(
                    Q_net,
                    optimizer,
                    lr_scheduler,
                    loss,
                    episode,
                    med_length,
                )

            length, solution = self.helper.total_distance(solution, W)
            if debug:
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
            self.path_lengths.append(length)
            self.writer.add_scalar("Pathlength", length, episode)
            self.__createTensorboardLogs(Q_net, episode)
            if trial is not None:
                if len(self.losses) < 1:
                    trial.report(-1.0, episode)
                else:
                    trial.report(
                        np.median(
                            np.convolve(
                                np.array(self.losses),
                                np.ones((100,)) / 100,
                                mode="valid",
                            )
                        ),
                        episode,
                    )
                if trial.should_prune():
                    self.writer.close()
                    raise optuna.exceptions.TrialPruned()

        return np.median(
            np.convolve(np.array(self.losses), np.ones((100,)) / 100, mode="valid")
        )

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

        # plt.figure(figsize=(8, 5))
        # plt.scatter(range(len(self.rewards)), self.rewards, "-o")
        # plt.ylabel("Rewards")
        # plt.xlabel("training iteration")

        plt.figure(figsize=(8, 5))
        plt.plot(_moving_avg(self.path_lengths, 100))
        plt.ylabel("average length")
        plt.xlabel("episode")
        plt.show()

    def getBestOder(
        self,
        samples: list = False,
        plot: bool = False,
        numCarts: int = 6,
        sampleReqs: list = False,
        modelFolder: Path = "",
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

        if modelFolder == "":
            modelFolder = self.folder_name

        all_lengths_fnames = [f for f in os.listdir(modelFolder) if f.endswith(".tar")]
        shortest_fname = sorted(
            all_lengths_fnames, key=lambda s: float(s.split(".tar")[0].split("_")[-1])
        )[0]
        fname = shortest_fname.split("_")
        emb = int(fname[fname.index("emb") + 1])
        it = int(fname[fname.index("it") + 1])

        Q_func, Q_net, optimizer, lr_scheduler = self.init_model(
            os.path.join(modelFolder, shortest_fname),
            EMBEDDING_DIMENSIONS=emb,
            EMBEDDING_ITERATIONS_T=it,
            debug=plot,
        )
        best_solution = {}
        best_value = float("inf")

        coords, W_np, _ = self.getData(samples=samples, sampleReqsList=sampleReqs)
        clist = []
        for c in coords:
            clist.append(c[4])
        max_len = np.max([len(a) for a in clist])

        components = np.asarray(
            [
                np.pad(a, (0, max_len - len(a)), "constant", constant_values=-1)
                for a in clist
            ],
            dtype=np.int32,
        )
        components = torch.from_numpy(components)
        self.helper = UtilFunctions(components)
        W = torch.tensor(
            W_np, dtype=torch.float, requires_grad=False, device=self.device
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
                "components": components,
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
    torch.multiprocessing.set_start_method("spawn")
    START_TIME = time.perf_counter()
    EMBEDDING_DIMENSIONS = 16
    EMBEDDING_ITERATIONS_T = 4
    runmodel = RunModel(
        dbpath=r"C:\Users\stephan.schumacher\Documents\repos\dh-reinforcement-optimization\data\SMD_Material_Stueli.txt",
        numSamples=24,
        tuning=False,
        allowDuplicates=False,
    )
    # exit()
    # runmodel.generateData()

    # all_lengths_fnames = [
    #     f for f in os.listdir(runmodel.folder_name) if f.endswith(".tar")
    # ]
    # shortest_fname = sorted(
    #     all_lengths_fnames, key=lambda s: float(s.split(".tar")[0].split("_")[-1])
    # )[0]
    # fname = shortest_fname.split("_")
    # emb = int(fname[fname.index("emb") + 1])
    # it = int(fname[fname.index("it") + 1])

    Q_Function, QNet, Adam, ExponentialLR = runmodel.init_model(
        # fname=os.path.join(runmodel.folder_name, shortest_fname),
        EMBEDDING_DIMENSIONS=EMBEDDING_DIMENSIONS,
        EMBEDDING_ITERATIONS_T=EMBEDDING_ITERATIONS_T,
        OPTIMIZER=torch.optim.Adam,
    )
    runmodel.fit(
        Q_func=Q_Function,
        Q_net=QNet,
        optimizer=Adam,
        lr_scheduler=ExponentialLR,
        NR_EPISODES=500,  # die sind noch über, bestes modell laden
        MIN_EPSILON=0.7,
        EPSILON_DECAY_RATE=6e-4,
        N_STEP_QL=4,
        BATCH_SIZE=16,
        GAMMA=0.7,
        debug=True,
    )
    runmodel.plotMetrics()

    END_TIME = time.perf_counter() - START_TIME
    exit()
    path = Path(os.getcwd() + os.path.normpath("/2days.xlsx"))

    loader = KappaLoader(
        path,
        dbpath=r"C:\Users\stephan.schumacher\Documents\repos\dh-reinforcement-optimization\data\SMD_Material_Stueli.txt",
    )

    # samples, sampleReqs = loader()
    runmodel = RunModel(
        dbpath=r"C:\Users\stephan.schumacher\Documents\repos\dh-reinforcement-optimization\data\SMD_Material_Stueli.txt",
        numSamples=300,
        tuning=False,
        allowDuplicates=True,
        overwriteDevice="cpu",
    )

    best_value, best_solution = runmodel.getBestOder(plot=True, numCarts=3)
    print(f"This run took {END_TIME} seconds | {END_TIME / 60} Minutes")
    validate = Validate(
        best_value,
        best_solution,
        dbpath=r"C:\Users\stephan.schumacher\Documents\repos\dh-reinforcement-optimization\data\SMD_Material_Stueli.txt",
        calcGroups=False,
        overlapThreshhold=0.5,
    )
    validate.plotSoltions()
