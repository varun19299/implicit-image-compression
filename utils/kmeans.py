import math
import torch
from time import time
import numpy as np
from dataclasses import dataclass

from matplotlib import pyplot as plt

from typing import Union


@dataclass
class KMeans:
    """
    Kmeans clustering algorithm implemented with PyTorch

    Parameters:
      n_clusters: int,
        Number of clusters

      max_iter: int, default: 100
        Maximum number of iterations

      tol: float, default: 0.0001
        Tolerance

      verbose: int, default: 0
        Verbosity

      mode: {'euclidean', 'cosine'}, default: 'euclidean'
        Type of distance measure

    Attributes:
      cluster_centers_: torch.Tensor, shape: [n_clusters, n_features]
    """

    n_clusters: int
    init: Union[str, torch.Tensor] = "random"
    max_iter: int = 100
    tol: float = 1e-4
    verbose: int = 0
    mode: str = "euclidean"

    def __post_init__(self):
        try:
            import PYNVML

            self._pynvml_exist = True
        except ModuleNotFoundError:
            self._pynvml_exist = False

    @staticmethod
    def cos_sim(a, b):
        """
        Compute cosine similarity of 2 sets of vectors

        Parameters:
        a: torch.Tensor, shape: [m, n_features]

        b: torch.Tensor, shape: [n, n_features]
        """
        a_norm = a.norm(dim=-1, keepdim=True)
        b_norm = b.norm(dim=-1, keepdim=True)
        a = a / (a_norm + 1e-8)
        b = b / (b_norm + 1e-8)
        return a @ b.transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        """
        Compute euclidean similarity of 2 sets of vectors

        Parameters:
        a: torch.Tensor, shape: [m, n_features]

        b: torch.Tensor, shape: [n, n_features]
        """
        return (
            2 * a @ b.transpose(-2, -1)
            - (a ** 2).sum(dim=1)[..., :, None]
            - (b ** 2).sum(dim=1)[..., None, :]
        )

    def remaining_memory(self):
        """
        Get remaining memory in gpu
        """
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if self._pynvml_exist:
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            remaining = info.free
        else:
            remaining = torch.cuda.memory_allocated()
        return remaining

    def max_sim(self, a, b):
        """
        Compute maximum similarity (or minimum distance) of each vector
        in a with all of the vectors in b

        Parameters:
        a: torch.Tensor, shape: [m, n_features]

        b: torch.Tensor, shape: [n, n_features]
        """
        device = a.device.type
        batch_size = a.shape[0]

        if self.mode == "cosine":
            sim_func = self.cos_sim
        elif self.mode == "euclidean":
            sim_func = self.euc_sim

        if device == "cpu":
            sim = sim_func(a, b)
            max_sim_v, max_sim_i = sim.max(dim=-1)
            return max_sim_v, max_sim_i
        else:
            if a.dtype == torch.float:
                # 32 bits or 4 bytes per float (full precision)
                expected_memory = a.shape[0] * a.shape[1] * b.shape[0] * 4
            elif a.dtype == torch.half:
                # 32 bits or 4 bytes per float (half precision)
                expected_memory = a.shape[0] * a.shape[1] * b.shape[0] * 2

            ratio = math.ceil(expected_memory / self.remaining_memory())
            subbatch_size = math.ceil(batch_size / ratio)
            msv, msi = [], []

            for i in range(ratio):
                if i * subbatch_size >= batch_size:
                    continue
                sub_x = a[i * subbatch_size : (i + 1) * subbatch_size]
                sub_sim = sim_func(sub_x, b)
                sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
                del sub_sim
                msv.append(sub_max_sim_v)
                msi.append(sub_max_sim_i)
            if ratio == 1:
                max_sim_v, max_sim_i = msv[0], msi[0]
            else:
                max_sim_v = torch.cat(msv, dim=0)
                max_sim_i = torch.cat(msi, dim=0)
            return max_sim_v, max_sim_i

    def fit_predict(self, X):
        """
        Combination of fit() and predict() methods.
        This is faster than calling fit() and predict() seperately.

        Parameters:
        X: torch.Tensor, shape: [n_samples, n_features]

        cluster_centers_: {torch.Tensor, None}, default: None
          if given, cluster_centers_ will be initialized with given tensor
          if None, cluster_centers_ will be randomly chosen from X

        Return:
        labels: torch.Tensor, shape: [n_samples]
        """
        batch_size, emb_dim = X.shape
        device = X.device.type

        if self.init == "random":
            self.cluster_centers_ = X[
                np.random.choice(batch_size, size=[self.n_clusters], replace=False)
            ]
        elif isinstance(self.init, torch.Tensor):
            n_samp, n_feat = X.shape
            assert self.init.shape == (
                self.n_clusters,
                n_feat,
            ), f"Shape mismatch in init, expected {(self.n_clusters, n_feat)}, got {self.init.shape}"
            self.cluster_centers_ = self.init

        else:
            print("Currently only random and guess init allowed")
            raise

        num_points_in_clusters = torch.ones(self.n_clusters, device=device)
        closest = None
        for i in range(self.max_iter):
            iter_time = time()
            x = X
            closest = self.max_sim(a=x, b=self.cluster_centers_)[1]
            matched_clusters, counts = closest.unique(return_counts=True)

            c_grad = torch.zeros_like(self.cluster_centers_)
            expanded_closest = closest[None].expand(len(matched_clusters), -1)
            mask = (expanded_closest == matched_clusters[:, None]).float()
            c_grad[matched_clusters] = mask @ x / mask.sum(-1)[..., :, None]

            error = (c_grad - self.cluster_centers_).pow(2).sum()

            num_points_in_clusters[matched_clusters] += counts
            self.cluster_centers_ = c_grad
            if self.verbose >= 2:
                print(
                    "iter:",
                    i,
                    "error:",
                    error.item(),
                    "time spent:",
                    round(time() - iter_time, 4),
                )
            if error <= self.tol:
                break

        return closest

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to

        Parameters:
        X: torch.Tensor, shape: [n_samples, n_features]

        Return:
        labels: torch.Tensor, shape: [n_samples]
        """
        return self.max_sim(a=X, b=self.cluster_centers_)[1]

    def fit(self, X):
        """
        Perform kmeans clustering

        Parameters:
        X: torch.Tensor, shape: [n_samples, n_features]
        """
        self.fit_predict(X)
        return self

    def fit_append_0_predict(self, X):
        """
        Perform kmeans clustering
        Add 0 to the centroid
        Predict

        Parameters:
        X: torch.Tensor, shape: [n_samples, n_features]
        """
        self.fit_predict(X)
        prepend = torch.zeros_like(self.cluster_centers_)[:1]
        self.cluster_centers_ = torch.cat((prepend, self.cluster_centers_))
        return self.predict(X)
