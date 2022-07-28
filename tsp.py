import numpy as np
import torch
import random
import math
from collections import namedtuple
import os
import time

from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

from scipy.signal import medfilt
import shutil

""" Note: the code is not optimized for GPU
"""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

START_TIME = time.perf_counter()

# shutil.rmtree(Path(os.getcwd() + "/models"))


def get_graph_mat(n=10, size=1):
    """Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
    Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
    """
    coords = size * np.random.uniform(size=(n, 2))
    dist_mat = distance_matrix(coords, coords)
    return coords, dist_mat


def plot_graph(coords, mat):
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


from pathlib import Path
import itertools
from src.misc.dataloader import DataLoader
import random

# coords, W_np = get_graph_mat(n=20)
path = Path(os.getcwd() + os.path.normpath("/data"))

products = {}
for i in os.listdir(path):
    product = i
    overallLen = 0
    allComponents = []
    for m in ["m10", "m20"]:
        data_path = path / i / m

        dataloader = DataLoader(data_path)
        data, components, offsets = dataloader()

        overallLen += len(data) * len(offsets)
        allComponents.append(components["Component Code"].unique())

    products[product] = {
        "len": overallLen,
        "comps": list(itertools.chain.from_iterable(allComponents)),
    }


def getData(key=False):
    def compare(p1, p2):
        l2 = len(list(set(p1) & set(p2)))
        l1 = len(p1)
        return l2 / l1

    globalList = {}
    for i in products.keys():
        currentList = products.copy()
        currentList.pop(i)
        # currentDict = {i: {"overlap": 1, "len": products[i]["len"]}}
        currentDict = [[products[i]["len"] / 1000, 1]]
        for j in currentList:
            currentDict.append(
                [
                    products[j]["len"] / 1000,
                    compare(products[i]["comps"], products[j]["comps"]),
                ]
            )
        globalList[i] = currentDict

    npArray = []
    if key == False:
        product, coords = random.choice(list(globalList.items()))
    else:
        coords = globalList[key]
    for i in coords:
        npArray.append(i)

    coords = np.asarray(npArray)
    W_np = distance_matrix(coords, coords)
    return coords, W_np


coords, W_np = getData()
W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=device)
print(coords.shape, W_np.shape, W.shape)
coords, W_np = get_graph_mat(n=18)
W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=device)
print(coords.shape, W_np.shape, W.shape)
plot_graph(coords, W_np)
plt.show()

State = namedtuple("State", ("W", "coords", "partial_solution"))


def state2tens(state):
    """Creates a Pytorch tensor representing the history of visited nodes, from a (single) state tuple.

    Returns a (Nx5) tensor, where for each node we store whether this node is in the sequence,
    whether it is first or last, and its (x,y) coordinates.
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
        ]
        for i in range(nr_nodes)
    ]

    return torch.tensor(xv, dtype=torch.float32, requires_grad=False, device=device)


class QNet(nn.Module):
    """The neural net that will parameterize the function Q(s, a)

    The input is the state (containing the graph and visited nodes),
    and the output is a vector of size N containing Q(s, a) for each of the N actions a.
    """

    def __init__(self, emb_dim, T=4):
        """emb_dim: embedding dimension p
        T: number of iterations for the graph embedding
        """
        super(QNet, self).__init__()
        self.emb_dim = emb_dim
        self.T = T

        # We use 5 dimensions for representing the nodes' states:
        # * A binary variable indicating whether the node has been visited
        # * A binary variable indicating whether the node is the first of the visited sequence
        # * A binary variable indicating whether the node is the last of the visited sequence
        # * The (x, y) coordinates of the node.
        self.node_dim = 5

        # We can have an extra layer after theta_1 (for the sake of example to make the network deeper)
        nr_extra_layers_1 = 4

        # Build the learnable affine maps:
        self.theta1 = nn.Linear(self.node_dim, self.emb_dim, True)
        self.theta2 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta3 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta4 = nn.Linear(1, self.emb_dim, True)
        self.theta5 = nn.Linear(2 * self.emb_dim, 1, True)
        self.theta6 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta7 = nn.Linear(self.emb_dim, self.emb_dim, True)

        self.theta1_extras = [
            nn.Linear(self.emb_dim, self.emb_dim, True)
            for _ in range(nr_extra_layers_1)
        ]

    def forward(self, xv, Ws):
        # xv: The node features (batch_size, num_nodes, node_dim)
        # Ws: The graphs (batch_size, num_nodes, num_nodes)

        num_nodes = xv.shape[1]
        batch_size = xv.shape[0]

        # pre-compute 1-0 connection matrices masks (batch_size, num_nodes, num_nodes)
        conn_matrices = torch.where(
            Ws > 0, torch.ones_like(Ws), torch.zeros_like(Ws)
        ).to(device)

        # Graph embedding
        # Note: we first compute s1 and s3 once, as they are not dependent on mu
        mu = torch.zeros(batch_size, num_nodes, self.emb_dim, device=device)
        s1 = self.theta1(xv)  # (batch_size, num_nodes, emb_dim)
        for layer in self.theta1_extras:
            s1 = layer(F.relu(s1))  # we apply the extra layer

        s3_1 = F.relu(
            self.theta4(Ws.unsqueeze(3))
        )  # (batch_size, nr_nodes, nr_nodes, emb_dim) - each "weigth" is a p-dim vector
        s3_2 = torch.sum(
            s3_1, dim=1
        )  # (batch_size, nr_nodes, emb_dim) - the embedding for each node
        s3 = self.theta3(s3_2)  # (batch_size, nr_nodes, emb_dim)

        for t in range(self.T):
            s2 = self.theta2(conn_matrices.matmul(mu))
            mu = F.relu(s1 + s2 + s3)

        """ prediction
        """
        # we repeat the global state (summed over nodes) for each node,
        # in order to concatenate it to local states later
        global_state = self.theta6(
            torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1)
        )

        local_action = self.theta7(mu)  # (batch_dim, nr_nodes, emb_dim)

        out = F.relu(torch.cat([global_state, local_action], dim=2))
        return self.theta5(out).squeeze(dim=2)


""" See what the model returns
"""
model = QNet(3, T=1).to(device)
# coords, W_np = get_graph_mat(n=10)
coords, W_np = getData()
W = torch.tensor(W_np, dtype=torch.float32, device=device)
xv = torch.rand((1, W.shape[0], 5)).to(device)  # random node state
Ws = W.unsqueeze(0)

y = model(xv, Ws)
print("model output: {}".format(y))


class QFunction:
    def __init__(self, model, optimizer, lr_scheduler):
        self.model = model  # The actual QNet
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = nn.MSELoss()

    def predict(self, state_tsr, W):
        # batch of 1 - only called at inference time
        with torch.no_grad():
            estimated_rewards = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0))
        return estimated_rewards[0]

    def get_best_action(self, state_tsr, state):
        """Computes the best (greedy) action to take from a given state
        Returns a tuple containing the ID of the next node and the corresponding estimated reward
        """
        W = state.W
        estimated_rewards = self.predict(state_tsr, W)  # size (nr_nodes,)
        sorted_reward_idx = estimated_rewards.argsort(descending=True)

        solution = state.partial_solution

        already_in = set(solution)
        sorted_reward_idx_list = sorted_reward_idx.tolist()
        for idx in sorted_reward_idx_list:
            if (
                len(solution) == 0 or W[solution[-1], idx] > 0
            ) and idx not in already_in:
                return idx, estimated_rewards[idx].item()
        # return (0, 0)

    def batch_update(self, states_tsrs, Ws, actions, targets):
        """Take a gradient step using the loss computed on a batch of (states, Ws, actions, targets)

        states_tsrs: list of (single) state tensors
        Ws: list of W tensors
        actions: list of actions taken
        targets: list of targets (resulting estimated rewards after taking the actions)
        """
        Ws_tsr = torch.stack(Ws).to(device)
        xv = torch.stack(states_tsrs).to(device)
        self.optimizer.zero_grad()

        # the rewards estimated by Q for the given actions
        estimated_rewards = self.model(xv, Ws_tsr)[range(len(actions)), actions]

        loss = self.loss_fn(estimated_rewards, torch.tensor(targets, device=device))
        loss_val = loss.item()

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss_val


# Note: we store state tensors in experience to compute these tensors only once later on
Experience = namedtuple(
    "Experience",
    ("state", "state_tsr", "action", "reward", "next_state", "next_state_tsr"),
)


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.nr_inserts = 0

    def remember(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        self.nr_inserts += 1

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return min(self.nr_inserts, self.capacity)


def total_distance(solution, W):
    if len(solution) < 2:
        return 0  # there is no travel

    total_dist = 0
    for i in range(len(solution) - 1):
        x = W[solution[i], solution[i + 1]]
        total_dist += W[solution[i], solution[i + 1]].item()

    # if this solution is "complete", go back to initial point
    if len(solution) == W.shape[0]:
        total_dist += W[solution[-1], solution[0]].item()

    return total_dist


def is_state_final(state):
    return len(set(state.partial_solution)) == state.W.shape[0]


def get_next_neighbor_random(state):
    solution, W = state.partial_solution, state.W

    if len(solution) == 0:
        return random.choice(range(W.shape[0]))
    already_in = set(solution)
    candidates = list(
        filter(lambda n: n.item() not in already_in, W[solution[-1]].nonzero())
    )
    if len(candidates) == 0:
        return None
    return random.choice(candidates).item()


SEED = 1  # A seed for the random number generator

# Graph
NR_NODES = 18  # Number of nodes N
EMBEDDING_DIMENSIONS = 10  # Embedding dimension D
EMBEDDING_ITERATIONS_T = 2  # Number of embedding iterations T

# Learning
NR_EPISODES = 6001
MEMORY_CAPACITY = 10000
N_STEP_QL = 4  # Number of steps (n) in n-step Q-learning to wait before computing target reward estimate
BATCH_SIZE = 32

GAMMA = 0.7
INIT_LR = 3e-3
LR_DECAY_RATE = 1.0 - 2e-5  # learning rate decay

MIN_EPSILON = 0.1
EPSILON_DECAY_RATE = 6e-4  # epsilon decay

FOLDER_NAME = "./models"  # where to checkpoint the best models


def init_model(fname=None):
    """Create a new model. If fname is defined, load the model from the specified file."""
    Q_net = QNet(EMBEDDING_DIMENSIONS, T=EMBEDDING_ITERATIONS_T).to(device)
    optimizer = optim.Adam(Q_net.parameters(), lr=INIT_LR)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_RATE)

    if fname is not None:
        checkpoint = torch.load(fname)
        Q_net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    Q_func = QFunction(Q_net, optimizer, lr_scheduler)
    return Q_func, Q_net, optimizer, lr_scheduler


def checkpoint_model(model, optimizer, lr_scheduler, loss, episode, avg_length):
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)

    fname = os.path.join(FOLDER_NAME, "ep_{}".format(episode))
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


# seed everything for reproducible results first:
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Create module, optimizer, LR scheduler, and Q-function
Q_func, Q_net, optimizer, lr_scheduler = init_model()

# Create memory
memory = Memory(MEMORY_CAPACITY)

# Storing metrics about training:
found_solutions = dict()  # episode --> (coords, W, solution)
losses = []
path_lengths = []

# keep track of median path length for model checkpointing
current_min_med_length = float("inf")

for episode in range(NR_EPISODES):
    # sample a new random graph
    coords, W_np = get_graph_mat(n=NR_NODES)
    # coords, W_np = getData()
    W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=device)

    # current partial solution - a list of node index
    solution = [random.randint(0, NR_NODES - 1)]

    # current state (tuple and tensor)
    current_state = State(partial_solution=solution, W=W, coords=coords)
    current_state_tsr = state2tens(current_state)

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
    while not is_state_final(current_state):
        t += 1  # time step of this episode

        if epsilon >= random.random():
            # explore
            next_node = get_next_neighbor_random(current_state)
            nr_explores += 1
        else:
            # exploit
            next_node, est_reward = Q_func.get_best_action(
                current_state_tsr, current_state
            )
            if episode % 50 == 0:
                print(
                    "Ep {} | current sol: {} / next est reward: {}".format(
                        episode, solution, est_reward
                    )
                )

        next_solution = solution + [next_node]

        # reward observed for taking this step
        reward = -(total_distance(next_solution, W) - total_distance(solution, W))

        next_state = State(partial_solution=next_solution, W=W, coords=coords)
        next_state_tsr = state2tens(next_state)

        # store rewards and states obtained along this episode:
        states.append(next_state)
        states_tsrs.append(next_state_tsr)
        rewards.append(reward)
        actions.append(next_node)

        # store our experience in memory, using n-step Q-learning:
        if len(solution) >= N_STEP_QL:
            memory.remember(
                Experience(
                    state=states[-N_STEP_QL],
                    state_tsr=states_tsrs[-N_STEP_QL],
                    action=actions[-N_STEP_QL],
                    reward=sum(rewards[-N_STEP_QL:]),
                    next_state=next_state,
                    next_state_tsr=next_state_tsr,
                )
            )

        if is_state_final(next_state):
            for n in range(1, N_STEP_QL):
                memory.remember(
                    Experience(
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
        if len(memory) >= BATCH_SIZE and len(memory) >= 2000:
            experiences = memory.sample_batch(BATCH_SIZE)

            batch_states_tsrs = [e.state_tsr for e in experiences]
            batch_Ws = [e.state.W for e in experiences]
            batch_actions = [e.action for e in experiences]
            batch_targets = []

            for i, experience in enumerate(experiences):
                target = experience.reward
                if not is_state_final(experience.next_state):
                    _, best_reward = Q_func.get_best_action(
                        experience.next_state_tsr, experience.next_state
                    )
                    target += GAMMA * best_reward
                batch_targets.append(target)

            print("batch targets: {}".format(batch_targets))
            loss = Q_func.batch_update(
                batch_states_tsrs, batch_Ws, batch_actions, batch_targets
            )
            losses.append(loss)

            """ Save model when we reach a new low average path length
            """
            med_length = np.median(path_lengths[-100:])
            if med_length < current_min_med_length:
                current_min_med_length = med_length
                checkpoint_model(
                    Q_net, optimizer, lr_scheduler, loss, episode, med_length
                )

    length = total_distance(solution, W)
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
        found_solutions[episode] = (W.clone(), coords.copy(), [n for n in solution])


def _moving_avg(x, N=10):
    return np.convolve(np.array(x), np.ones((N,)) / N, mode="valid")


plt.figure(figsize=(8, 5))
plt.semilogy(_moving_avg(losses, 100))
plt.ylabel("loss")
plt.xlabel("training iteration")

plt.figure(figsize=(8, 5))
plt.plot(_moving_avg(path_lengths, 100))
plt.ylabel("average length")
plt.xlabel("episode")
plt.show()

""" Get file with smallest distance
"""
all_lengths_fnames = [f for f in os.listdir(FOLDER_NAME) if f.endswith(".tar")]
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
Q_func, Q_net, optimizer, lr_scheduler = init_model(
    os.path.join(FOLDER_NAME, shortest_fname)
)

""" A function to plot solutions
"""


def plot_solution(coords, mat, solution):
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


END_TIME = time.perf_counter() - START_TIME
print(f"This run took {END_TIME} seconds | {END_TIME / 60} Minutes")
"""Generate example solutions"""
NR_NODES = 20

from sklearn import preprocessing

"""for key in globalList:
    npArray = []
    for i in globalList[key]:
        npArray.append(i)

    coords = np.asarray(npArray)
    
    W_np = distance_matrix(coords, coords)

    W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=device)

    solution = [random.randint(0, coords.shape[0] - 1)]
    current_state = State(partial_solution=solution, W=W, coords=coords)
    current_state_tsr = state2tens(current_state)

    while not is_state_final(current_state):
        # next_node, est_reward = Q_func.get_best_action(current_state_tsr, current_state)
        returnValues = Q_func.get_best_action(current_state_tsr, current_state)
        if returnValues == None:
            continue
        solution = solution + [returnValues[0]]
        current_state = State(partial_solution=solution, W=W, coords=coords)
        current_state_tsr = state2tens(current_state)

    plt.figure()
    print(solution)
    plot_solution(coords, W, solution)
    plt.title("model / len = {}".format(total_distance(solution, W)))

    # for comparison, plot a random solution
    plt.figure()
    random_solution = list(range(NR_NODES))
    plot_solution(coords, W, random_solution)
    plt.title("random / len = {}".format(total_distance(random_solution, W)))
    plt.show()"""

runs = 5

best_solution = {}
best_value = 10000

for key in products.keys():
    #
    coords, W_np = getData(key)
    # plot_graph(coords, 1)
    # plt.show()
    # coords, W_np = get_graph_mat(n=NR_NODES)
    W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=device)

    solution = [random.randint(0, coords.shape[0] - 1)]
    current_state = State(partial_solution=solution, W=W, coords=coords)
    current_state_tsr = state2tens(current_state)

    while not is_state_final(current_state):
        next_node, est_reward = Q_func.get_best_action(current_state_tsr, current_state)

        solution = solution + [next_node]
        current_state = State(partial_solution=solution, W=W, coords=coords)
        current_state_tsr = state2tens(current_state)

    if total_distance(solution, W) < best_value:
        best_value = total_distance(solution, W)
        best_solution = {"W": W, "solution": solution, "coords": coords}

plt.figure()
print(
    "The best value for this iteration is: ",
    total_distance(best_solution["solution"], best_solution["W"]),
)
plot_solution(best_solution["coords"], best_solution["W"], best_solution["solution"])
plt.title(
    "model / len = {}".format(
        total_distance(best_solution["solution"], best_solution["W"])
    )
)
plt.figure()
random_solution = list(range(best_solution["coords"].shape[0]))
plot_solution(best_solution["coords"], best_solution["W"], random_solution)
plt.title(
    "random / len = {}".format(total_distance(random_solution, best_solution["W"]))
)
plt.show()
"""plt.figure()
print(solution)
plot_solution(coords, W, solution)
plt.title("model / len = {}".format(total_distance(solution, W)))

# for comparison, plot a random solution
plt.figure()
random_solution = list(range(coords.shape[0]))
plot_solution(coords, W, random_solution)
plt.title("random / len = {}".format(total_distance(random_solution, W)))
plt.show()"""
