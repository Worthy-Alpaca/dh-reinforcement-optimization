import random
import torch
import torch.nn as nn


class QFunction:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        lr_scheduler,
        device=torch.device("cpu"),
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = nn.MSELoss()
        self.device = device

    def predict(self, state_tsr, W):
        with torch.no_grad():
            estimated_rewards = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0))
        return estimated_rewards[0]

    def get_best_action(self, state_tsr, state):
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
        print("here")
        return None, 0

    def batch_update(self, states_tsrs, Ws, actions, targets):
        Ws_tsr = torch.stack(Ws).to(self.device)
        xv = torch.stack(states_tsrs).to(self.device)
        self.optimizer.zero_grad()

        estimated_rewards = self.model(xv, Ws_tsr)[range(len(actions)), actions]

        loss = self.loss_fn(
            estimated_rewards, torch.tensor(targets, device=self.device)
        )
        loss_val = loss.item()

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss_val


class Memory(object):
    def __init__(self, capacity=10000) -> None:
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


class UtilFunctions:
    def __init__(self, coords) -> None:
        self.coords = coords

    def is_state_final(self, state):
        return len(set(state.partial_solution)) == state.W.shape[0]

    def total_distance(self, solution, W):
        if len(solution) < 2:
            return 0

        total_dist = 0

        def compare(p1, p2):
            l2 = len(list(set(p1) & set(p2)))
            l1 = len(p1)
            return l2 / l1

        for i in range(len(solution) - 1):
            l2 = len(
                list(
                    set(self.coords[solution[i], 3])
                    & set(self.coords[solution[i + 1], 3])
                )
            )
            l1 = len(self.coords[solution[i], 3])
            total_dist += l2 / l1

        """for i in range(len(solution) - 1):
            if solution[i] == None or solution[i + 1] == None:
                continue
            x = W[solution[i], solution[i + 1]]
            total_dist += W[solution[i], solution[i + 1]].item()

        if len(solution) == W.shape[0]:
            total_dist += W[solution[-1], solution[0]].item()"""

        return total_dist

    def get_next_neighbor_random(self, state):
        # replace this with overlap calculations
        solution, W = state.partial_solution, state.W

        if len(solution) == 0:
            return random.choice(range(W.shape[0]))
        already_in = set(solution)
        candidates = list(
            filter(lambda n: n.item() not in already_in, W[solution[-1]].nonzero())
        )

        if len(candidates) == 0:
            return random.choice(range(W.shape[0]))
        return random.choice(candidates).item()
