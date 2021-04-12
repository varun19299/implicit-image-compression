"""
File and Buffer handling utils

Taken from pytorch: https://github.com/pytorch/pytorch/blob/master/torch/serialization.py
"""

import sys
import pathlib
import io


def _check_seekable(f) -> bool:
    def raise_err_msg(patterns, e):
        for p in patterns:
            if p in str(e):
                msg = (
                    str(e)
                    + ". You can only load from a file that is seekable."
                    + " Please pre-load the data into a buffer like io.BytesIO and"
                    + " try to load from it instead."
                )
                raise type(e)(msg)
        raise e

    try:
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        raise_err_msg(["seek", "tell"], e)
    return False


def _is_path(name_or_buffer):
    return isinstance(name_or_buffer, str) or (
        sys.version_info[0] == 3 and isinstance(name_or_buffer, pathlib.Path)
    )


class _opener(object):
    def __init__(self, file_like):
        self.file_like = file_like

    def __enter__(self):
        return self.file_like

    def __exit__(self, *args):
        pass


class _open_file(_opener):
    def __init__(self, name, mode):
        super(_open_file, self).__init__(open(name, mode))

    def __exit__(self, *args):
        self.file_like.close()


class _open_buffer_reader(_opener):
    def __init__(self, buffer):
        super(_open_buffer_reader, self).__init__(buffer)
        _check_seekable(buffer)


class _open_buffer_writer(_opener):
    def __exit__(self, *args):
        self.file_like.flush()


def open_file_like(name_or_buffer, mode) -> _opener:
    """
    Open string and pathlib paths via open
    and
    treat other handlers as such

    :param name_or_buffer:
    :param mode: r/ w/ rb/ wb etc.
    :return: file handler
    """
    if _is_path(name_or_buffer):
        return _open_file(name_or_buffer, mode)
    else:
        if "w" in mode:
            return _open_buffer_writer(name_or_buffer)
        elif "r" in mode:
            return _open_buffer_reader(name_or_buffer)
        else:
            raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")
