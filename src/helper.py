import math
import random
import torch
import torch.nn as nn
from types import FunctionType


class QFunction:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        lr_scheduler: FunctionType,
        loss_fn: torch.nn = nn.HuberLoss,
        device=torch.device("cpu"),
    ) -> None:
        """Initiate the Q Function

        Args:
            model (nn.Module): The current model instance.
            optimizer (torch.optim): The current optimizer instance.
            lr_scheduler (FunctionType): The current learning rate scheduler
            loss_fn (torch.nn, optional): The current loss function. Defaults to nn.HuberLoss.
            device (torch.device or str, optional): The current device. Defaults to torch.device("cpu").
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn()
        self.device = device

    def predict(self, state_tsr: torch.tensor, W: torch.tensor):
        """Method to predict a reward.

        Args:
            state_tsr (torch.tensor): The current state tensor.
            W (torch.tensor): The current distance matrix tensor.

        Returns:
            torch.tensor: The predicted rewards tensor.
        """
        with torch.no_grad():
            estimated_rewards = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0))
        return estimated_rewards[0]

    def get_best_action(self, state_tsr, state):
        """Method to get the current best action.

        Args:
            state_tsr (torch.tensor): The current state tensor.
            state (State): The current state.

        Returns:
            tuple: index and estimated reward of best action.
        """
        W = state.W

        estimated_rewards = self.predict(state_tsr, W)  # size (nr_nodes,)

        sorted_reward_idx = estimated_rewards.argsort(descending=True)
        solution = state.partial_solution

        already_in = set(solution)
        sorted_reward_idx_list = sorted_reward_idx.tolist()
        for idx in sorted_reward_idx_list:
            if (
                len(solution) == 0 or W[solution[-1], idx] >= 0
            ) and idx not in already_in:
                return idx, estimated_rewards[idx].item()
        return 0, 0

    def batch_update(self, states_tsrs, Ws, actions, targets):
        """Method to calculate the batch loss.

        Args:
            states_tsrs (torch.tensor): The current state tensor.
            Ws (torch.tensor): The current distance matrix tensor.
            actions (torch.tensor): The current batch actions.
            targets (torch.tensor): The current batch targets.

        Returns:
            float: The calculated batch loss.
        """
        Ws_tsr = torch.stack(Ws).to(self.device)
        xv = torch.stack(states_tsrs).to(self.device)
        self.optimizer.zero_grad(set_to_none=True)

        if None in actions:
            for i in actions:
                for x in range(len(actions)):
                    if x in actions:
                        continue
                    else:
                        actions[i] = x

        with torch.cuda.amp.autocast():
            estimated_rewards = self.model(xv, Ws_tsr)[range(len(actions)), actions]

            loss = self.loss_fn(
                estimated_rewards, torch.tensor(targets, device=self.device)
            )
        loss_val = loss.item()
        if torch.any(torch.isnan(loss)):
            print()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss_val


class Memory(object):
    def __init__(self, capacity=10000) -> None:
        """Class that represents a memory.

        Args:
            capacity (int, optional): The memory capacity. Defaults to 10000.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.nr_inserts = 0

    def remember(self, experience):
        """Method to remember the current experience

        Args:
            experience (torch.tensor): The current experience.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        self.nr_inserts += 1

    def sample_batch(self, batch_size):
        """Method to return a sample batch.

        Args:
            batch_size (int): The needed batch size.

        Returns:
            torch.tensor: The created batch.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return min(self.nr_inserts, self.capacity)


class UtilFunctions:
    def __init__(self, coords) -> None:
        """Initiate the utility function collection.

        Args:
            coords (torch.tensor): The current components.
        """
        self.coords = coords

    def is_state_final(self, state):
        return len(set(state.partial_solution)) == state.W.shape[0]

    def calc_total_time(self, solution: list):
        """Calculates the total time for manufacturing and preparation.

        Args:
            solution (list): The final solution list.

        Returns:
            float: The calculated total time.
        """
        total_overlap = 0
        r1 = 0
        t1 = 0
        for step in range(len(solution) - 1):
            idx1, idx2 = solution[step], solution[step + 1]
            c1 = self.coords[idx1]
            c2 = self.coords[idx2]
            a_cat_b, counts = torch.cat([c1, c2]).unique(return_counts=True)
            overlapComponents = a_cat_b[torch.where(counts.gt(1))]
            r1 += Cartsetup(c1)
            total_overlap += Cartsetup(overlapComponents)
            del c1, c2
        return total_overlap, t1

    def total_distance(self, solution: list, W):
        """Method to calculate the total distance of a given solution.

        Args:
            solution (list): The current solution.
            W (torch.tensor): The current distance matrix tensor.

        Returns:
            tuple: calculated distance, solution
        """
        if len(solution) < 2:
            return 0, solution

        total_dist = 0.0

        for i in range(len(solution) - 1):
            idx1, idx2 = solution[i], solution[i + 1]

            running_dist = W[idx1, idx2].item()

            l1 = self.coords[solution[i]]
            l2 = self.coords[solution[i + 1]]
            a_cat_b, counts = torch.cat([l1, l2]).unique(return_counts=True)
            l2 = a_cat_b[torch.where(counts.gt(1))]

            if l2.shape[0] == 1 and l2[0].item() == -1:
                overlap = 1e-2
            else:
                overlap = l2.shape[0] / l1.shape[0]
            del l1, l2
            total_dist += running_dist / overlap

        if len(solution) == W.shape[0]:
            total_dist += W[solution[-1], solution[0]].item()

        return total_dist, solution

    def get_next_neighbor_random(self, state):
        """Method to get a random action for a given state (explore).

        Args:
            state (State): The current state.

        Returns:
            int: The random index of a all available nodes.
        """
        solution, W = state.partial_solution, state.W

        if len(solution) == 0:
            return random.choice(range(W.shape[0]))
        already_in = set(solution)
        candidates = list(
            filter(lambda n: n.item() not in already_in, W[solution[-1] - 1].nonzero())
        )

        if len(candidates) == 0:
            return random.choice(range(W.shape[0]))
        return random.choice(candidates).item()


def Cartsetup(comps: list):
    """Function to calculate setup time for a given list of components.

    Args:
        comps (list): The current component list.

    Returns:
        float: The calculated time.
    """
    time = 0

    complexity = 36 / len(comps)
    for i in range(len(comps)):
        time = ((60 + random.randint(0, 30)) * complexity + 9.8) + time
    return time


def Coating(Ymax) -> float:
    """simulates the time for coating a PCB.

    Args:
        Ymax (int): The maximum y value on a given PCB.

    Returns:
        float: The calculated time.
    """
    velocity = 20  # mm/s

    # highest coordinate on PCB
    return math.sqrt(0**2 + Ymax**2) / velocity


class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        """Class to redirect a console interaction to a given TKinter widget.

        Args:
            widget (tk.widget): The widget which should display the console interaction
            tag (str, optional): The given tag. Defaults to "stdout".
        """
        self.widget = widget
        self.tag = tag

    def write(self, str):
        """Method to write to the widget.

        Args:
            str (str): The string to be displayed.
        """
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")
