from dataclasses import dataclass
import numpy as np
from typing import BinaryIO
import functools


def reset_bytes(f):
    @functools.wraps(f)
    def foo(self):
        written_bytes = f(self)
        self._written_bytes = 0
        return written_bytes

    return foo


@dataclass()
class NumpyParser:
    handler: BinaryIO

    def __post_init__(self):
        self._written_bytes = 0
        for method in ["read"]:
            setattr(self, method, getattr(self.handler, method))

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __enter__(self):
        return self

    def write(self, array: np.ndarray) -> int:
        self._written_bytes += self.handler.write(array.tobytes())
        return self._written_bytes

    @reset_bytes
    def flush(self):
        self.handler.flush()
        return self._written_bytes


def test_plain_handler():
    import numpy as np

    array = np.random.rand(3, 3)
    # Write
    with open("/tmp/test_plain_handler.txt", "wb") as opened_handler:
        with NumpyParser(opened_handler) as compressor:
            compressor.write(array)
            bytes_written = compressor.flush()
            assert (
                compressor._written_bytes == 0
            ), "written_bytes not zeroed after flush"

    # Read
    with open("/tmp/test_plain_handler.txt", "rb") as opened_handler:
        with NumpyParser(opened_handler) as decompressor:
            bytes = decompressor.read()
            read_array = np.frombuffer(
                bytes, dtype=array.dtype, count=array.size
            ).reshape(*array.shape)
            assert (read_array == array).all(), "Array not recovered"
            assert bytes_written == read_array.nbytes, "Bytes mismatch"


if __name__ == "__main__":
    test_plain_handler()
