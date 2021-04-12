import zstandard as zstd
import torch
from torch import nn
from typing import BinaryIO

from utils import file_handler
from encoding import utils as encoding_utils
from collections import OrderedDict
from scipy.sparse import csc_matrix

from zipfile import ZipFile
from pathlib import Path
from typing import Union
import json


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
    cctx = zstd.ZstdCompressor(level=level)

    model = model.to(torch.device("cpu"))
    state_dict = model.state_dict()
    meta_data = OrderedDict()

    if isinstance(file_name, str):
        file_name = Path(file_name)

    binary_file_name = file_name.parent / f"{file_name.stem}.data"

    with file_handler.open_file_like(binary_file_name, "wb") as opened_handler:
        with cctx.stream_writer(opened_handler) as compressor:
            for e, (name, tensor) in enumerate(state_dict.items()):
                array = tensor.numpy()

                # Convert to CSC only if
                # sparsity crosses 50%
                if encoding_utils.sparsity(array) > 0.5:
                    sparse_array = csc_matrix(array, dtype=array.dtype)
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
                    array = tensor.numpy()
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


def decompress_state_dict(model: nn.Module, file_handler: BinaryIO):
    pass
