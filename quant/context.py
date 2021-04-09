from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import nn

from quant.kmeans_quantize import KmeansQuant


@dataclass
class Quantize:
    model: nn.Module
    optim: torch.optim.Optimizer
    quant_conf: DictConfig

    def __enter__(self):
        prepare_method = getattr(self, f"_prepare_{self.quant_conf.name}")
        prepare_method()
        return self

    def __exit__(self, type, value, traceback):
        pass

    def convert(self):
        convert_method = getattr(self, f"_convert_{self.quant_conf.name}")
        return convert_method()

    def _convert_QAT(self):
        return torch.quantization.convert(self.model.eval().cpu())

    def _convert_KMeans(self):
        return self.compress.update_weights()

    def _prepare_QAT(self):
        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'fbgemm' for server inference and
        # 'qnnpack' for mobile inference. Other quantization configurations such
        # as selecting symmetric or assymetric quantization and MinMax or L2Norm
        # calibration techniques can be specified here.
        self.model.qconfig = torch.quantization.get_default_qat_qconfig(
            self.quant_conf.qconfig
        )

        # Prepare the model for QAT. This inserts observers and fake_quants in
        # the model that will observe weight and activation tensors during calibration.
        torch.quantization.prepare_qat(self.model, inplace=True)

    def _prepare_KMeans(self):
        skip_ll = self.quant_conf.get("skip_ll", ["layers.0.linear", "layers.7.linear"])
        self.compress = KmeansQuant(
            self.model, self.optim, bits=self.quant_conf.bits, skip_ll=skip_ll
        )
