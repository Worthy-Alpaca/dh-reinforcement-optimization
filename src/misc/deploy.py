import torch
import joblib
import os
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class DeplotmentException(Exception):
    pass


class DeployModel:
    def __init__(self, path: Path) -> None:
        """Initiate the deployment of the Neural Network.

        Args:
            path (Path): Path to the saved NN Parameters.

        Raises:
            DeplotmentException: If the Save Folder doesn't exist.
            DeplotmentException: If something in the Model Loading process went wrong. Will notify of the error.
        """
        if not os.path.exists(path):
            raise DeplotmentException(
                "The Model cannot be deployed from the given Path. Make sure that the Path is correct"
            )
        try:
            self.scaleX: MinMaxScaler = joblib.load(path / "scaleStateX.gz")
            self.scaleY: MinMaxScaler = joblib.load(path / "scaleStateY.gz")
            self.model: torch.nn.Module = torch.jit.load(path / "modelState.pt").to(
                "cpu"
            )
        except Exception as e:
            raise DeplotmentException(f"The Loading of the Model failed. {e}")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Method to predict with the loaded Neural Network.

        Args:
            data (np.ndarray): The Data needed for the prediction.

        Returns:
            np.ndarray: The predicted value(s).
        """
        data = data.reshape(1, -1)
        try:
            data = self.scaleX.transform(data)
        except:
            pass
        data = torch.from_numpy(data)
        prediction = self.model(data.float())
        prediction = prediction.cpu().detach().numpy()
        try:
            prediction = self.scaleY.inverse_transform(prediction)
        except:
            pass
        return prediction
