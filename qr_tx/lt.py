import io
import math
import random
from dataclasses import dataclass
from typing import List

import numpy as np

ROBUST_FAILURE_PROBABILITY = 0.01


@dataclass
class Symbol:
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

    m = n // 2 + 1
    r = n / m

    probabilities = [0, 1 / n]
    probabilities += [1 / (k * (k - 1)) for k in range(2, n + 1)]

    robust_addl = [0] + [1 / (i * m) for i in range(1, m)]
    robust_addl += [math.log(r / ROBUST_FAILURE_PROBABILITY) / m]
    robust_addl += [0 for k in range(m + 1, n + 1)]

    probabilities = np.add(robust_addl, probabilities)
    probabilities /= np.sum(probabilities)

    return probabilities


def get_block_degrees(n: int, k: int):
    """Returns the random degrees from a given distribution of probabilities.
    The degrees distribution must look like a Poisson distribution and the
    degree of the first drop is 1 to ensure the start of decoding.
    """

    probabilities = robust_distribution(n)

    population = list(range(0, n + 1))
    return [1] + random.choices(population, probabilities, k=k - 1)


def get_block_indicies(idx: int, degree: int, n_blocks: int) -> List[int]:
    random.seed(idx)
    return random.sample(range(n_blocks), degree)


def prepare_data(data: bytes, block_sz: int) -> List[np.array]:
    n_blocks = len(data) // block_sz
    if len(data) % block_sz != 0:
        n_blocks += 1
    chunks = [data[i : i + block_sz] for i in range(0, len(data), block_sz)]
    if len(chunks[-1]) != block_sz:
        chunks[-1] = chunks[-1] + bytearray(block_sz - len(chunks[-1]))
    return [Symbol.data_from_bytes(d) for d in chunks]


def encode(data: bytes, block_sz: int, redundancy: float) -> List[Symbol]:
    blocks = prepare_data(data, block_sz)
    n_blocks = len(blocks)
    n_total_bytes = len(data)
    print(f"total bytes: {n_total_bytes}")
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


def decode(symbols: List[Symbol]):

    n_blocks = symbols[0].n_blocks
    blocks = [None] * n_blocks
    n_total_bytes = symbols[0].n_total_bytes
    symbol_src_idxs = {
        s.idx: set(get_block_indicies(s.idx, s.degree, n_blocks)) for s in symbols
    }

    progress_made = True

    while progress_made:
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

        for symbol in symbols:
            solved_srcs = symbol_src_idxs[symbol.idx] & sovled_blocks
            if symbol.degree > 1 and len(solved_srcs):
                for solved_src in solved_srcs:
                    symbol.data = np.bitwise_xor(blocks[solved_src], symbol.data)
                    symbol.degree -= 1
                symbol_src_idxs[symbol.idx] -= solved_srcs

        symbols = [s for s in symbols if s.degree > 0]

    n_missing = len([b for b in blocks if b is None])
    if n_missing:
        print(f"Not enough symbols to recover data, missing: {n_missing}")
        return None
    blocks = [bytes(b) for b in blocks]
    block_sz = len(blocks[0])
    if n_total_bytes % block_sz != 0:
        blocks[-1] = blocks[-1][: n_total_bytes % block_sz]
    buffer = io.BytesIO()
    for block in blocks:
        buffer.write(block)
    buffer.seek(0)

    return buffer.read()
