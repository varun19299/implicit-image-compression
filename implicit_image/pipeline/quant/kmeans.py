from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from implicit_image.pipeline.quant.kmeans_helper import kmeans_fit, kmeans_predict


@dataclass
class KmeansQuant:
    """
    Implements quantization technique used in

    "DEEP COMPRESSION: COMPRESSING DEEP NEURAL NETWORKS WITH
    PRUNING, TRAINED QUANTIZATION AND HUFFMAN CODING",

    by Han et al., ICLR 2016s.

    * K Mean clustering for quantization

    Idea:

    1. Forward hook: compute clusters for required layers, apply them to weights
    2. Backward hook: Use grad to update cluster positions
    """

    model: nn.Module
    optim: torch.optim.Optimizer

    # k means clusters = 2**bits
    bits: int = 5

    skip_ll: List = field(
        default_factory=lambda: ["layers.0.linear", "layers.7.linear"]
    )

    def __post_init__(self):
        self.forward_pre_hook_ll = []
        self.backward_hook_ll = []

        for name, module in self.model.named_modules():
            if name in self.skip_ll:
                continue
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
        # kmeans_cluster
        centroids, labeled_weight, new_weight = self.find_centroids(module)

        module.labeled_weight = labeled_weight
        module.centroids = centroids
        module.weight.data = new_weight

    def update_weights(self):
        # Remove all hooks
        for hook in [*self.forward_pre_hook_ll, *self.backward_hook_ll]:
            hook.remove()

        for layer, (name, module) in enumerate(self.model.named_modules()):
            if name in self.skip_ll:
                continue
            if isinstance(module, nn.Linear):
                centroids = module.centroids
                labeled_weight = module.labeled_weight

                # Sort
                # _, indices = torch.sort(torch.abs(centroids))
                # centroids = centroids[indices]
                # labeled_weight = indices[labeled_weight]

                # Set as params
                module.centroids = nn.Parameter(centroids, requires_grad=False)
                module.labeled_weight = nn.Parameter(
                    labeled_weight, requires_grad=False
                )

                # Drop centroids into labels
                new_weight = self.labels_to_weights(labeled_weight, centroids)
                module.weight.data = new_weight

                # del module.labeled_weight

    def remove_labels_centroids(self):
        for layer, (name, module) in enumerate(self.model.named_modules()):
            if name in self.skip_ll:
                continue
            if isinstance(module, nn.Linear):
                del module.labeled_weight
                del module.centroids

    def find_centroids(self, module: nn.Module):
        weight = module.weight.data
        device = weight.device
        shape = weight.shape
        dtype = weight.dtype

        # rehape
        weight = weight.reshape(-1, 1)

        # Exclude zeros
        weight_nonzero = weight[weight != 0].reshape(-1, 1)

        # Linear guess
        guess = torch.linspace(
            weight_nonzero.min(),
            weight_nonzero.max(),
            self.n_clusters - 1,
            device=device,
            dtype=dtype,
        ).reshape(-1, 1)

        labels, centroids = kmeans_fit(
            weight_nonzero, num_clusters=self.n_clusters - 1, cluster_centers=guess
        )

        # Append 0.0 as a centroid
        prepend = torch.zeros_like(centroids)[:1]
        centroids = torch.cat((prepend, centroids))

        # Keep only unique centroids
        centroids = torch.unique(centroids)
        _, indices = torch.sort(centroids.abs())
        centroids = rearrange(centroids[indices], "n -> n 1")

        labels = kmeans_predict(weight, centroids).reshape(*shape)

        # Drop centroids into labels
        centroids = rearrange(centroids, "n 1-> n")
        new_weight = self.labels_to_weights(labels, centroids)

        return centroids, labels, new_weight

    def labels_to_weights(self, labeled_weight, centroids):
        """
        Copy centroids to respective labels on labeled_weight

        :param labeled_weight: labeled weight
        :param centroids: code book
        :return:
        """
        new_weight = centroids[labeled_weight]
        return new_weight

    def get_centroids_gradients(self, grad_input, labeled_weight, centroids):
        dw = torch.zeros_like(centroids)

        w_grad = grad_input[2].t().data
        w_grad = torch.flatten(w_grad)
        labeled_weight = torch.flatten(labeled_weight)

        dw.scatter_add_(0, labeled_weight, w_grad)

        return dw

    def scalar_quantization(self, module, grad_input, grad_output):
        # TODO: Is this even needed? Centroids are re-computed in the forward pass.
        labeled_weight = module.labeled_weight
        centroids = module.centroids
        dw = self.get_centroids_gradients(grad_input, labeled_weight, centroids)

        # Gradient update
        module.centroids = centroids - self.learning_rate * dw


if __name__ == "__main__":
    from implicit_image.models.siren import Siren
    from implicit_image.utils.timer import catchtime

    model = Siren()
    in_shape = (5, 5, 2)
    n_iters = 5

    if torch.cuda.is_available():
        model = model.cuda()

    def _loop(model, in_shape, n_iters: int = 5, backward: bool = False):
        for _ in range(n_iters):
            in_tensor = torch.rand(*in_shape)
            if torch.cuda.is_available():
                in_tensor = in_tensor.cuda()

            if backward:
                model(in_tensor).sum().backward()
            else:
                model(in_tensor)

    print(f"Reporting average of {n_iters} runs each.")

    # w/o kmeans
    with catchtime() as t:
        _loop(model, in_shape, n_iters)
    print(f"\nTiming without kMeans {t() / n_iters}")

    with catchtime() as t:
        _loop(model, in_shape, n_iters, backward=True)
    print(f"Backward without kMeans {t() / n_iters}")

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    comp = KmeansQuant(model, optim)

    # w kmeans
    with catchtime() as t:
        _loop(model, in_shape, n_iters)
    print(f"\nTiming with kMeans {t() / n_iters}")

    with catchtime() as t:
        _loop(model, in_shape, n_iters, backward=True)
    print(f"Backward with kMeans {t() / n_iters}")

    comp.update_weights()
    comp.remove_labels_centroids()
