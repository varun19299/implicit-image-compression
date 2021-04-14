import zstandard
import torch
from torch import nn
import numpy as np

from implicit_image.utils import file_handler
from implicit_image.pipeline.entropy_coding import utils as encoding_utils
from collections import OrderedDict
from scipy.sparse import csc_matrix

from zipfile import ZipFile
from pathlib import Path
from typing import Dict, Union
import json


def linear_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only linear layers

    If quantized, store labels + centroids
    :param model:
    :return:
    """
    state_dict = model.cpu().state_dict()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "centroids") and hasattr(module, "labeled_weight"):
                state_dict.pop(f"{name}.weight")
                state_dict[f"{name}.labeled_weight"] = state_dict[
                    f"{name}.labeled_weight"
                ].to(torch.uint8)

    return state_dict


def compress_state_dict(
    model: nn.Module, file_name: Union[str, Path], level: int = 22
) -> int:
    """
    Workflow

    1. Decide if model's weights are sparse or dense
    2.

    :param model:
    :param file:
    :param level:
    :return:
    """
    cctx = zstandard.ZstdCompressor(level=level)

    state_dict = linear_state_dict(model)
    meta_data = OrderedDict()

    if isinstance(file_name, str):
        file_name = Path(file_name)

    binary_file_name = file_name.parent / f"{file_name.stem}.data"

    with file_handler.open_file_like(binary_file_name, "wb") as opened_handler:
        with cctx.stream_writer(opened_handler, write_size=1000000) as compressor:
            for e, (name, tensor) in enumerate(state_dict.items()):
                array = tensor.numpy()
                print(name, array.dtype)

                # Convert to CSC only if
                # sparsity crosses 60%
                if encoding_utils.sparsity(array) > 0.6:
                    sparse_array = csc_matrix(array, dtype=array.dtype)
                    sparse_array.indices = sparse_array.indices.astype(np.uint8)
                    sparse_array.indptr = sparse_array.indptr.astype(np.uint8)

                    for attribute in ["data", "indices", "indptr"]:
                        sparse_rep = getattr(sparse_array, attribute)
                        compressor.write(sparse_rep)

                        info_dict = {
                            "shape": sparse_rep.shape,
                            "dtype": str(sparse_rep.dtype),
                            "order": e,
                        }
                        meta_data[f"{name}_{attribute}"] = info_dict
                else:
                    compressor.write(array)
                    info_dict = {
                        "shape": array.shape,
                        "dtype": str(array.dtype),
                        "order": e,
                    }
                    meta_data[name] = info_dict

            # Flush compressor, get bytes written
            compressed_bytes = compressor.flush()

    meta_data_file = f"{file_name.stem}_meta_data.json"

    # Write meta-data and binary file into a zipfile
    with ZipFile(file_name, "w") as zf:
        zf.write(binary_file_name)
        with zf.open(meta_data_file, "w") as f:
            f.write(json.dumps(meta_data, indent=2).encode("utf-8"))

    binary_file_name.unlink(missing_ok=True)

    return compressed_bytes


def decompress_state_dict(model: nn.Module, file_name: Union[str, Path]):
    pass
