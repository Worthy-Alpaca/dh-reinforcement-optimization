import random
import torch
import math
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
        self.loss_fn = nn.L1Loss()
        self.device = device

    def predict(self, state_tsr, W):
        with torch.no_grad():
            estimated_rewards = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0))
        return estimated_rewards[0]

    def get_best_action(self, state_tsr, state):
        W = state.W
        if torch.any(torch.isnan(W)):
            print()
        if torch.any(torch.isnan(state_tsr)):
            print()

        estimated_rewards = self.predict(state_tsr, W)  # size (nr_nodes,)

        if torch.any(torch.isnan(estimated_rewards)):
            print()
            estimated_rewards = torch.nan_to_num(estimated_rewards)
            print()
        sorted_reward_idx = estimated_rewards.argsort(descending=True)
        solution = state.partial_solution

        already_in = set(solution)
        sorted_reward_idx_list = sorted_reward_idx.tolist()
        for idx in sorted_reward_idx_list:
            x = W[solution[-1], idx]
            if (
                len(solution) == 0 or W[solution[-1], idx] >= 0
            ) and idx not in already_in:
                return idx, estimated_rewards[idx].item()
        # print("here get best action")
        return 0, 0

    def batch_update(self, states_tsrs, Ws, actions, targets):
        Ws_tsr = torch.stack(Ws).to(self.device)
        xv = torch.stack(states_tsrs).to(self.device)
        self.optimizer.zero_grad()

        if None in actions:
            for i in actions:
                for x in range(len(actions)):
                    if x in actions:
                        continue
                    else:
                        actions[i] = x

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
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

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

    def calc_total_time(self, solution: list):
        """Calculates the total time for manufacturing and preparation.

        Args:
            solution (list): The final solution list.

        Returns:
            float: The calculated total time.
        """
        total_time = 0
        total_overlap = 0
        r1 = 0
        t1 = 0
        for step in range(len(solution) - 1):
            # MAYBE ALSO CALCULATE THE OVERLAP TO THE NEXT NEXT PRODUCT
            idx1, idx2 = solution[step], solution[step + 1]
            c1 = self.coords[idx1, 4]
            c2 = self.coords[idx2, 4]
            overlapComponents = list(set(c1) & set(c2))
            r1 += Cartsetup(c1)
            total_overlap += Cartsetup(overlapComponents)
            t1 += self.coords[idx1, 1] * self.coords[idx1, 6]
            # total_time += r1 + self.coords[idx1, 1] * self.coords[idx1, 6]
        total_time = t1 + r1
        return total_overlap, t1

    def total_distance(self, solution: list, W):
        if len(solution) < 2:
            return 0, solution

        total_dist = 0.0

        for i in range(len(solution) - 1):
            idx1, idx2 = solution[i], solution[i + 1]
            # if idx2 == None:
            #     for x in range(W.shape[0]):
            #         if x in solution:
            #             continue
            #         else:
            #             idx2 = x
            #             solution[i + 1] = x
            # x = W[idx1, idx2]
            running_dist = W[idx1, idx2].item()
            # REPLACE CONSTANT WITH FRACTION FOR PROGRAM CHANGES
            # ADD REMAINDER TO GROUP CALCULATION
            overlapComponents = list(
                set(self.coords[idx1, 4]) & set(self.coords[idx2, 3])
            )
            l1 = self.coords[solution[i], 4]
            l2 = self.coords[solution[i + 1], 4]
            # total_dist += self.coords[solution[i], 1]
            # total_dist -= Cartsetup(overlapComponents)
            l2 = list(set(l1) & set(l2))
            overlap = len(l2) / len(l1)
            if overlap == 0:
                total_dist += running_dist / 1e-8
            else:
                total_dist += running_dist / overlap
            # total_dist += l2 / l1
            SETUPMINUTES = 20
            # total_dist += 60 * SETUPMINUTES

        if len(solution) == W.shape[0]:
            total_dist += W[solution[-1], solution[0]].item()

        return total_dist, solution

    def get_next_neighbor_random(self, state):
        # replace this with overlap calculations
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
    time = 0

    # print(f"Setting up Cart {cart} with {len(self.feedcart[key])} components")
    complexity = len(comps) / 36
    for i in range(len(comps)):
        time = (60 + random.randint(0, 0) * complexity + 9.8) + time
    return time


def Coating(Ymax) -> float:
    """simulates the time for coating a PCB

    Returns:
        float: The calculated time.
    """
    velocity = 20  # mm/s

    # highest coordinate on PCB
    return math.sqrt(0**2 + Ymax**2) / velocity
