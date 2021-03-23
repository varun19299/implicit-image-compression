"""Some helper functions for FeatherMap, including:
    - get_block_rows: Get complete rows from range within matrix
"""
from typing import List


def get_block_rows(i1: int, j1: int, i2: int, j2: int, n: int) -> List[int]:
    """Return range of full (complete) rows from an (n x n) matrix, starting from row, col
    [i1, j1] and ending at [i2, j2]. E.g.

    | _ x x x |            | x x x |
    | x x x x |  ------>             + | x x x x |
    | x x x x |                        | x x x x | +
    | x x _ _ |                                      | x x |

    Necessary to make the most use of vectorized matrix multiplication. Sequentially
    calculating V[i, j] = V1[i, :] @ V2[:, j] leads to large latency.
    """
    row = []
    # Handle all cases for j1=0
    if j1 == 0:
        # All row(s) complete
        if j2 == n - 1:
            row.extend([i1, i2 + 1])
            return row
        # First row complete, last row incomplete
        elif i2 > i1:
            row.extend([i1, i2])
            return row
        # First row incomplete (from right), no additional rows
        else:
            return row
    # First row incomplete (from left), last row complete
    if j2 == n - 1 and i2 > i1:
        row.extend([i1 + 1, i2 + 1])
        return row
    # First row incomplete, last row incomplete; has at least one full row
    if i2 - i1 > 1:
        row.extend([i1 + 1, i2])
        return row
    # First row incomplete, second row incomplete
    return row
