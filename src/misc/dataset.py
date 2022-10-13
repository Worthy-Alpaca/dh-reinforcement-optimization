import numpy as np
import torch
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset, DataLoader


class ProductDataset(Dataset):
    def __init__(self, data: np.ndarray, clist: np.ndarray) -> None:
        """Class that contains all data used for model training.

        Args:
            data (np.ndarray): The data points.
            clist (np.ndarray): The corresponding component arrays.
        """
        super().__init__()
        self.data = torch.from_numpy(data[:, :3].astype(np.float32))
        self.components = torch.from_numpy(clist)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.components[index]
        return x, self.data[index]


class ProductDataloader(DataLoader):
    """Class to load the data."""

    def __iter__(self):
        for batch in super().__iter__():
            yield batch[1], distance_matrix(batch[1], batch[1]), batch[0]
