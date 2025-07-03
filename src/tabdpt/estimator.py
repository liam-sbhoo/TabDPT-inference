import json
import logging
import os
from pathlib import Path

import gdown
import numpy as np
import torch
from appdirs import user_cache_dir
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .model import TabDPTModel
from .utils import FAISS, convert_to_torch_tensor

# Constants for model caching and download
_MODEL_NAME = "tabdpt1_1.safetensors"
_CACHE_BASE = Path(user_cache_dir("tabdpt", "Layer6"))
_CACHE_BASE.mkdir(parents=True, exist_ok=True)
_GDRIVE_FILE_ID = "1ARFl7uQ6bwcpP9lTPqDv1_G0M3VDW3mI"
logger = logging.getLogger(__name__)


def _get_checkpoint(path: str | None = None) -> Path:
    """
    Resolve the safetensors checkpoint file:
    1. If `path` is provided, return it directly.
    2. Otherwise, look in the user cache dir and download via gdown if missing.
    """
    if path:
        return Path(path)
    safetensors_path = _CACHE_BASE / _MODEL_NAME
    if safetensors_path.exists():
        return safetensors_path
    pth_path = _CACHE_BASE / "tmp.pth"
    if not os.path.exists(pth_path):
        logger.info(f"Downloading checkpoint from Google Drive to {pth_path}")
        url = f"https://drive.google.com/uc?id={_GDRIVE_FILE_ID}"
        gdown.download(url, str(pth_path), quiet=False)

    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
    model_state = checkpoint["model"]
    cfg = checkpoint["cfg"]
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_json = json.dumps(cfg_dict)
    metadata = {"cfg": cfg_json}
    save_file(model_state, safetensors_path, metadata=metadata)
    logger.info(f"Successfully converted {pth_path} to {safetensors_path}")
    # Remove the pth file after conversion
    pth_path.unlink()
    return safetensors_path


class TabDPTEstimator(BaseEstimator):
    def __init__(
        self,
        mode: str,
        path: str = None,
        inf_batch_size: int = 512,
        device: str = None,
        use_flash: bool = True,
        compile: bool = True,
    ):
        self.mode = mode
        self.inf_batch_size = inf_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_flash = use_flash
        self.path = _get_checkpoint(path)

        with safe_open(self.path, framework="pt", device=self.device) as f:
            meta = f.metadata()
            cfg_dict = json.loads(meta["cfg"])
            cfg = OmegaConf.create(cfg_dict)
            model_state = {k: f.get_tensor(k) for k in f.keys()}

        cfg.env.device = self.device
        self.model = TabDPTModel.load(model_state=model_state, config=cfg, use_flash=use_flash)
        self.model.eval()

        self.max_features = self.model.num_features
        self.max_num_classes = self.model.n_out
        self.compile = compile
        assert self.mode in ["cls", "reg"], "mode must be 'cls' or 'reg'"

    def fit(self, X, y):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert X.ndim == 2, "X must be a 2D array"
        assert y.ndim == 1, "y must be a 1D array"

        self.imputer = SimpleImputer(strategy="mean")
        X = self.imputer.fit_transform(X)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.faiss_knn = FAISS(X)
        self.n_instances, self.n_features = X.shape
        self.X_train = X
        self.y_train = y
        self.is_fitted_ = True
        if self.compile:
            self.model = torch.compile(self.model)

    def _prepare_prediction(self, X: np.ndarray):
        check_is_fitted(self)
        self.X_test = self.imputer.transform(X)
        self.X_test = self.scaler.transform(self.X_test)
        train_x, train_y, test_x = (
            convert_to_torch_tensor(self.X_train).to(self.device).float(),
            convert_to_torch_tensor(self.y_train).to(self.device).float(),
            convert_to_torch_tensor(self.X_test).to(self.device).float(),
        )

        # Apply PCA optionally to reduce the number of features
        if self.n_features > self.max_features:
            _, _, self.V = torch.pca_lowrank(train_x, q=self.max_features)
            train_x = train_x @ self.V
            test_x = test_x @ self.V
        return train_x, train_y, test_x
