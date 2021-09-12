""" Luby Transform encoding - see https://en.wikipedia.org/wiki/Luby_transform_code"""
import io
import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

FAILURE_PROBABILITY = 0.01


@dataclass
class Symbol:
    """Container for encoded symbol data"""

    idx: int
    degree: int
    n_blocks: int
    n_total_bytes: int
    data: np.array

    def data_as_bytes(self) -> bytes:
        return bytes(self.data)

    @staticmethod
    def data_from_bytes(b: bytes) -> np.array:
        return np.frombuffer(b, dtype=np.uint64)


def robust_distribution(n: int):
    """Construct robust soliton distribution - see https://en.wikipedia.org/wiki/Soliton_distribution"""

    m = n // 2 + 1
    r = n / m

    probabilities = [0, 1 / n]
    probabilities += [1 / (k * (k - 1)) for k in range(2, n + 1)]

    robust_addl = [0] + [1 / (i * m) for i in range(1, m)]
    robust_addl += [math.log(r / FAILURE_PROBABILITY) / m]
    robust_addl += [0 for k in range(m + 1, n + 1)]

    probabilities = np.add(robust_addl, probabilities)
    probabilities /= np.sum(probabilities)

    return probabilities


def get_block_degrees(n: int, k: int) -> List[int]:
    """
    Given n source blocks encoded to k symbols, return the degree for each output symbol
    """

    probabilities = robust_distribution(n)
    rng = np.random.default_rng()
    # prefixed with degree 1 to help start decoding with small n
    return [1] + rng.choice(range(0, n + 1), size=k - 1, p=probabilities).tolist()


def get_block_indicies(idx: int, degree: int, n_blocks: int) -> List[int]:
    """
    Given a block idx and degree, "deterministicly" return which source block idxs are part of symbol.
    Found the idea to use random sample seeded by block idx somewhere along the research process.

    Note: Numpy makes no guarantee about version compatability.
    TODO: Upgrade to proper PRNG
    """
    rng = np.random.default_rng(idx)
    return rng.choice(range(n_blocks), degree, replace=False).tolist()


def prepare_data(data: bytes, block_sz: int) -> List[np.array]:
    """
    Slice raw data into numpy blocks and handle padding if necessary
    """
    n_blocks = len(data) // block_sz
    if len(data) % block_sz != 0:
        n_blocks += 1
    chunks = [data[i : i + block_sz] for i in range(0, len(data), block_sz)]
    if len(chunks[-1]) != block_sz:
        chunks[-1] = chunks[-1] + bytearray(block_sz - len(chunks[-1]))
    return [Symbol.data_from_bytes(d) for d in chunks]


def encode(data: bytes, block_sz: int, redundancy: float) -> List[Symbol]:
    """
    Transform raw input data to stream of symbols.
    """
    if block_sz % 8 != 0:
        raise ValueError("Arg block_sz must be multiple of 8.")
    blocks = prepare_data(data, block_sz)
    n_blocks = len(blocks)
    n_total_bytes = len(data)
    n_frames = int(n_blocks * redundancy)
    degrees = get_block_degrees(n_blocks, n_frames)
    symbols = []
    for idx, degree in enumerate(degrees):
        src_idxs = get_block_indicies(idx, degree, n_blocks)
        data = blocks[src_idxs[0]]
        for other_idx in src_idxs[1:]:
            data = np.bitwise_xor(data, blocks[other_idx])

        symbol = Symbol(idx, degree, n_blocks, n_total_bytes, data)
        symbols.append(symbol)
    return symbols


def decode(symbols: List[Symbol]) -> Optional[bytes]:
    """
    Given a collection of input symbols, attempt to decode message
    """

    if not len(symbols):
        raise ValueError("Must pass at least one input symbol")

    n_blocks = symbols[0].n_blocks
    blocks = [None] * n_blocks
    n_total_bytes = symbols[0].n_total_bytes
    symbol_src_idxs = {
        s.idx: set(get_block_indicies(s.idx, s.degree, n_blocks)) for s in symbols
    }

    progress_made = True

    # For every iteration we:
    # 1) Resolve any Symbols of degree 1 to the source block
    # 2) For all newly solved source blocks, xor out data and decrement degree from dependet symbols and
    while progress_made:

        # 1) Resolve any Symbols of degree 1 to the corresponding source block
        progress_made = False
        sovled_blocks = set()
        for symbol in symbols:
            if symbol.degree > 1:
                continue
            symbol.degree = 0
            assert len(symbol_src_idxs[symbol.idx]) == 1
            block_idx = symbol_src_idxs[symbol.idx].pop()

            if blocks[block_idx] is not None:
                continue

            progress_made = True
            blocks[block_idx] = symbol.data
            sovled_blocks.add(block_idx)

        # 2) For all newly solved source blocks, xor out data and decrement degree from dependet symbols
        for symbol in symbols:
            solved_srcs = symbol_src_idxs[symbol.idx] & sovled_blocks
            if symbol.degree > 1 and len(solved_srcs):
                for solved_src in solved_srcs:
                    symbol.data = np.bitwise_xor(blocks[solved_src], symbol.data)
                    symbol.degree -= 1
                symbol_src_idxs[symbol.idx] -= solved_srcs

        # 3) Filter out fully decoded symbols
        symbols = [s for s in symbols if s.degree > 0]

    n_missing = len([b for b in blocks if b is None])
    if n_missing:
        log.info(
            f"Not enough symbols to recover data, number of missing blocks: {n_missing}"
        )
        return None

    # reconstruct original binary data from symbols
    blocks = [bytes(b) for b in blocks]
    block_sz = len(blocks[0])

    # remove added padding
    if n_total_bytes % block_sz != 0:
        blocks[-1] = blocks[-1][: n_total_bytes % block_sz]
    buffer = io.BytesIO()
    for block in blocks:
        buffer.write(block)
    buffer.seek(0)
    return buffer.read()
