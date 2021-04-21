import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import zstandard
from scipy.sparse import csc_matrix
from torch import nn

from implicit_image.pipeline.entropy_coding import utils as encoding_utils
from implicit_image.pipeline.entropy_coding.parsers import NumpyParser, LZMAParser


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

                if f"{name}.labeled_weight" not in state_dict:
                    raise KeyError("Please run .update() before compressing weights")
                state_dict[f"{name}.labeled_weight"] = state_dict[
                    f"{name}.labeled_weight"
                ].to(torch.uint8)

    return state_dict


def get_stream_writer(name: str, **kwargs):
    if name == "zstd":
        return zstandard.ZstdCompressor(level=kwargs["level"]).stream_writer
    elif name == "plain":
        return NumpyParser
    elif name == "lzma":
        return LZMAParser
    elif name == "huffman":
        return
    else:
        raise NotImplementedError(f"stream writer {name} not found.")


def get_stream_reader(name: str, **kwargs):
    if name == "zstd":
        return zstandard.ZstdDecompressor().stream_reader
    elif name == "plain":
        return NumpyParser
    elif name == "lzma":
        return LZMAParser
    elif name == "huffman":
        return
    else:
        raise NotImplementedError(f"stream writer {name} not found.")


def compress_state_dict(
    model: nn.Module, dir_name: Union[str, Path], stream_name: str, **kwargs
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

    stream_writer = get_stream_writer(stream_name, **kwargs)

    state_dict = linear_state_dict(model)
    meta_data = OrderedDict()

    if isinstance(dir_name, str):
        dir_name = Path(dir_name)
        dir_name.mkdir(exist_ok=True, parents=True)

    binary_file_name = dir_name / f"compressed_weights.data"
    meta_data_file = dir_name / "meta_data.json"

    with open(binary_file_name, "wb") as opened_handler:
        with stream_writer(opened_handler) as compressor:
            order = 0
            for name, tensor in state_dict.items():
                array = tensor.numpy()
                # Convert to CSC only if
                # sparsity crosses 60%
                if encoding_utils.sparsity(array) > 0.6:
                    sparse_array = csc_matrix(array, dtype=array.dtype)

                    dtype = np.uint8
                    if array.shape[1] > 256:
                        dtype = np.uint16
                    sparse_array.indices = sparse_array.indices.astype(dtype)
                    sparse_array.indptr = sparse_array.indptr.astype(np.uint16)

                    for attribute in ["data", "indices", "indptr"]:
                        sparse_rep = getattr(sparse_array, attribute)
                        compressor.write(sparse_rep)

                        info_dict = {
                            "shape": sparse_rep.shape,
                            "dtype": str(sparse_rep.dtype),
                            "name": f"{name}_{attribute}",
                        }
                        meta_data[order] = info_dict

                        # Increment order
                        order += 1
                else:
                    compressor.write(array)
                    info_dict = {
                        "shape": array.shape,
                        "dtype": str(array.dtype),
                        "name": name,
                    }
                    meta_data[order] = info_dict

                    # Increment order
                    order += 1

            # Flush compressor, get bytes written
            compressed_bytes = compressor.flush()

    with open(meta_data_file, "w") as f:
        f.write(json.dumps(meta_data, indent=2, sort_keys=True))

    return compressed_bytes


def decompress_state_dict(dir_name: Union[str, Path], stream_name: str, **kwargs):
    stream_reader = get_stream_reader(stream_name, **kwargs)

    if isinstance(dir_name, str):
        dir_name = Path(dir_name)

    binary_file_name = dir_name / f"compressed_weights.data"
    meta_data_file = dir_name / "meta_data.json"

    state_dict = {}

    with open(meta_data_file, "r") as f:
        meta_data = json.load(f)

    meta_data = {int(k): v for k, v in meta_data.items()}

    with open(binary_file_name, "rb") as f:
        with stream_reader(f) as decompressor:
            dec = decompressor.read()

            # Start extracting one-by-one
            offset = 0
            for order in sorted(meta_data.keys()):
                # Find shape size
                array_shape = meta_data[order]["shape"]
                array_size = np.array(array_shape, dtype=int).prod()

                # Find dtype
                dtype = meta_data[order]["dtype"]
                # convert from string to np.dtype
                dtype = getattr(np, dtype)

                # Find name
                name = meta_data[order]["name"]

                array = np.frombuffer(dec, dtype=dtype, count=array_size, offset=offset)

                # Reshape array
                array.resize(*array_shape)

                # Add to state_dict
                state_dict[name] = array

                # Update offset
                offset += array_size * np.dtype(dtype).itemsize

    tensor_state_dict = {}
    # Convert csc matrices
    for name in state_dict.keys():
        if not (
            ("centroids" in name)
            or ("_data" in name)
            or ("_indices" in name)
            or ("_indptr" in name)
        ):
            tensor_state_dict[name] = torch.from_numpy(state_dict[name]).float()
        elif "data" in name:
            root_name = name.replace("_data", "")

            data = state_dict[f"{root_name}_data"]
            if not np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.int32)
            else:
                data = data.astype(np.float32)

            indices = state_dict[f"{root_name}_indices"].astype(np.int32)
            indptr = state_dict[f"{root_name}_indptr"].astype(np.int32)

            weight = csc_matrix((data, indices, indptr), dtype=data.dtype).todense()

            # Convert centroid, labels
            if "labeled_weight" in root_name:
                centroid_name = root_name.replace("labeled_weight", "centroids")
                weight = state_dict[centroid_name][weight]

            tensor_state_dict[
                root_name.replace("labeled_weight", "weight")
            ] = torch.from_numpy(weight).float()

    return tensor_state_dict


def test_compress_decompress():
    from implicit_image.models.siren import Siren
    import torch
    from implicit_image.pipeline.masking import Masking
    from implicit_image.pipeline.masking.funcs.decay import CosineDecay
    from implicit_image.utils.train_helper import train_epoch
    from implicit_image.pipeline.quant.kmeans import KmeansQuant

    model = Siren()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    img = torch.rand(1, 1, 3)
    grid = torch.rand(1, 1, 2)

    decay = CosineDecay()
    mask = Masking(optim, decay, input_size=(1, 1, 2))
    mask.add_module(model)

    train_epoch(model, optim, grid, img, mask=mask)

    compress = KmeansQuant(model, optim)
    train_epoch(model, optim, grid, img, mask=mask)
    compress.update_weights()

    original = model.state_dict()

    compress_state_dict(
        model.half(), "/tmp/compress_test", stream_name="zstd", level=22
    )

    decomp = decompress_state_dict("/tmp/compress_test", stream_name="zstd")


if __name__ == "__main__":
    test_compress_decompress()
