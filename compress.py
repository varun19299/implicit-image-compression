import torch
from sklearn.cluster import KMeans
from utils.kmeans import KMeans as KMeans_torch
import numpy as np
import torch.nn as nn

from dataclasses import dataclass
from einops import rearrange


@dataclass
class DeepCompressor:
    """
    Implements quantization technique used in

    "DEEP COMPRESSION: COMPRESSING DEEP NEURAL NETWORKS WITH
    PRUNING, TRAINED QUANTIZATION AND HUFFMAN CODING",

    by Han et al., ICLR 2016s.

    * K Mean clustering for quantization
    * Huffman encoding for entropy based lossless compression

    Idea:

    1. Forward hook: compute clusters for required layers, apply them to weights
    2. Backward hook: Use grad to update cluster positions
    """

    model: nn.Module
    optim: torch.optim.Optimizer

    # k means clusters = 2**bits
    bits: int = 5

    def __post_init__(self):
        self.forward_pre_hook_ll = []
        self.backward_hook_ll = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Before a forward
                forward_pre_hook = module.register_forward_pre_hook(
                    self.kmeans_modify_weight
                )
                self.forward_pre_hook_ll.append(forward_pre_hook)

                # After a backward
                backward_hook = module.register_backward_hook(self.scalar_quantization)
                self.backward_hook_ll.append(backward_hook)

    @property
    def n_clusters(self):
        return 2 ** self.bits

    @property
    def learning_rate(self):
        return self.optim.defaults["lr"]

    def kmeans_modify_weight(self, module, input):
        device = module.weight.data.device

        # kmeans_cluster
        centroids, labeled_weight, new_weight = self.find_centroids(module.weight)

        module.labeled_weight = labeled_weight
        module.centroids = centroids
        module.weight.data = new_weight

    def update_weights(self):
        # Remove all hooks
        for hook in [*self.forward_pre_hook_ll, *self.backward_hook_ll]:
            hook.remove()

        for layer, (name, module) in enumerate(self.model.named_modules()):
            weight = module.weight.data.cpu().numpy()
            device = weight.data.device

            centroids = module.centroids
            labeled_weight = module.labeled_weight

            # Drop centroids into labels
            new_weight = self.labels_to_weights(labeled_weight, centroids)
            new_weight = new_weight.reshape(*weight.shape, dtype=np.int8)

            module.weight = torch.from_numpy(new_weight).to(device)
            del module.labeled_weight

    def find_centroids(self, weight):
        device = weight.data.device
        weight_np = weight.data  # .cpu().numpy()
        weight_np = weight_np.reshape(-1, 1)

        # Exclude zeros
        weight_nonzero = weight_np[weight_np != 0].reshape(-1, 1)

        # Linear guess
        # guess = np.linspace(
        #     weight_nonzero.min(),
        #     weight_nonzero.max(),
        #     self.n_clusters - 1,
        #     dtype=np.float32,
        # )
        guess = torch.linspace(
            weight_nonzero.min(),
            weight_nonzero.max(),
            self.n_clusters - 1,
        )
        guess = guess.reshape(-1, 1)

        # kmeans = KMeans(
        #     n_clusters=self.n_clusters - 1, init=guess, random_state=0, n_init=1
        # ).fit(weight_nonzero)

        kmeans = KMeans_torch(
            n_clusters=self.n_clusters - 1,
            centroids=guess,
        ).fit(weight_nonzero)
        centroids = kmeans.cluster_centers_

        # Append 0.0 as a centroid
        centroids = np.append(0.0, centroids)
        kmeans.cluster_centers_ = rearrange(centroids, "n -> n 1").astype(np.float32)

        labels = kmeans.predict(weight_np)

        # Drop centroids into labels
        new_weight = self.labels_to_weights(labels, centroids)
        new_weight = new_weight.reshape(*weight.shape)
        new_weight = torch.from_numpy(new_weight).float().to(device)

        return centroids, labels, new_weight

    def labels_to_weights(self, labeled_weight, centroids):
        """
        Copy centroids to respective labels on labeled_weight

        :param labeled_weight: labeled weight
        :param centroids: code book
        :return:
        """
        new_weight = np.zeros(shape=labeled_weight.shape, dtype=np.float32)
        for index, label in enumerate(labeled_weight):
            new_weight[index] = centroids[label]
        return labeled_weight

    def get_centroids_gradients(self, grad_input, labeled_weight, centroids):
        dw = np.zeros(shape=centroids.shape, dtype=np.float32)
        w_grad = grad_input[2].t().data.cpu().numpy()
        grad_w = w_grad.reshape(-1, 1)
        for index, label in enumerate(labeled_weight):
            dw[label] += grad_w[index]
        return dw

    def scalar_quantization(self, module, grad_input, grad_output):
        labeled_weight = module.labeled_weight
        centroids = module.centroids
        dw = self.get_centroids_gradients(grad_input, labeled_weight, centroids)

        # Gradient update
        module.centroids = centroids - self.learning_rate * dw


if __name__ == "__main__":
    from models.siren import Siren
    from utils.timer import catchtime

    model = Siren()

    # w/o kmeans
    with catchtime() as t:
        model(torch.rand(5, 5, 2))
    print(f"Timing without kMeans {t()}")

    with catchtime() as t:
        model(torch.rand(5, 5, 2)).sum().backward()
    print(f"Backward without kMeans {t()}")

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    comp = DeepCompressor(model, optim)

    # w kmeans
    with catchtime() as t:
        model(torch.rand(5, 5, 2))
    print(f"\nTiming with kMeans {t()}")

    with catchtime() as t:
        model(torch.rand(5, 5, 2)).sum().backward()
    print(f"Backward with kMeans {t()}")
