from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from torch_scatter import scatter_mean
from torchtyping import TensorType


def pairwise_euclidean(
    tensor_A: TensorType["batch", "features", float],
    tensor_B: TensorType["batch", "features", float],
):
    tensor_A = rearrange(tensor_A, "n f -> n 1 f")
    tensor_B = rearrange(tensor_B, "m f -> 1 m f")

    dis = (tensor_A - tensor_B) ** 2.0

    # return N*M matrix for pairwise distance
    dis = dis.sum(dim=-1)
    return dis


def pairwise_cosine(
    tensor_A: TensorType["batch", "features", float],
    tensor_B: TensorType["batch", "features", float],
):
    tensor_A = rearrange(tensor_A, "n f -> n 1 f")
    tensor_B = rearrange(tensor_B, "m f -> 1 m f")

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = tensor_A / tensor_A.norm(dim=-1, keepdim=True)
    B_normalized = tensor_B / tensor_B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*M matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1)

    return cosine_dis


distance_func_dict = {"euclidean": pairwise_euclidean, "cosine": pairwise_cosine}


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans_fit(
    X: TensorType["batch", "features", float],
    num_clusters: int,
    distance: str = "euclidean",
    cluster_centers=[],
    tolerance: float = 1e-4,
    iter_limit: int = 5,
) -> Tuple:
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tolerance: (float) threshold [default: 0.0001]
    :param iter_limit: hard limit for max number of iterations
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    pairwise_distance_function = distance_func_dict[distance]

    if not X.dtype == torch.float:
        X = X.float()

    # initialize
    if not len(cluster_centers):
        cluster_centers = initialize(X, num_clusters)

    for iteration in range(iter_limit):
        dis = pairwise_distance_function(X, cluster_centers)

        labels = torch.argmin(dis, dim=1)
        new_cluster_centers = scatter_mean(X, labels, dim=0)

        center_shift = torch.sqrt(
            torch.sum((cluster_centers - new_cluster_centers) ** 2, dim=1)
        ).sum()
        cluster_centers = new_cluster_centers
        if center_shift ** 2 < tolerance:
            break

    return labels, cluster_centers


def kmeans_predict(
    X: TensorType["batch", "features", float],
    cluster_centers: TensorType["batch", "features", float],
    distance: str = "euclidean",
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :return: (torch.tensor) cluster ids
    """
    dis = distance_func_dict[distance](X, cluster_centers)
    labels = torch.argmin(dis, dim=1)

    return labels
