import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance_matrix


class QNetModel(nn.Module):
    def __init__(self, emb_dim, T=4, device=torch.device("cpu")) -> None:
        super(QNetModel, self).__init__()
        self.emb_dim = emb_dim
        self.T = T

        self.node_dim = 5

        self.device = device

        nr_extra_layers_1 = T

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

        num_nodes = xv.shape[1]
        batch_size = xv.shape[0]

        conn_matrices = torch.where(
            Ws > 0, torch.ones_like(Ws), torch.zeros_like(Ws)
        ).to(self.device)

        mu = torch.zeros(batch_size, num_nodes, self.emb_dim, device=self.device)
        s1 = self.theta1(xv)
        for layer in self.theta1_extras:
            s1 = layer(F.relu(s1))

        s3_1 = F.relu(self.theta4(Ws.unsqueeze(3)))
        s3_2 = torch.sum(s3_1, dim=1)
        s3 = self.theta3(s3_2)

        for t in range(self.T):
            s2 = self.theta2(conn_matrices.matmul(mu))
            mu = F.relu(s1 + s2 + s3)

        global_state = self.theta6(
            torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1)
        )

        local_action = self.theta7(mu)

        out = F.relu(torch.cat([global_state, local_action], dim=2))
        return self.theta5(out).squeeze(dim=2)
