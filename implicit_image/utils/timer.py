"""
Source: https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
"""

from time import perf_counter
from contextlib import contextmanager


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


if __name__ == "__main__":

    with catchtime() as t:
        import time

        time.sleep(1)

    print(f"Execution time: {t():.4f} secs")
