import random
import string

import numpy as np

from qr_tx.lt import decode, encode, get_block_degrees, robust_distribution


def test_robust_distribution():
    probabilities = robust_distribution(10)
    probabilities_sum = np.sum(probabilities)

    assert (
        probabilities_sum >= 1 - 1e-4 and probabilities_sum <= 1 + 1e-4
    ), "Not a std distribution"


def test_get_block_degrees():
    res = get_block_degrees(10, 20)
    print(res)


def test_encode_decode():
    data = "".join(random.choice(string.ascii_letters) for _ in range(2000)).encode(
        "utf-8"
    )
    symbols = encode(data, 24, 2.0)
    decoded = decode(symbols)
    assert decoded == data


def test_encode_decode_data_loss():
    data_sz = 200000

    data = "".join(random.choice(string.ascii_letters) for _ in range(data_sz)).encode(
        "utf-8"
    )
    symbols = encode(data, 64, 1.8)
    print(f"using {len(symbols)} symbols")
    symbols = random.sample(symbols, int(len(symbols) * 0.91))
    decoded = decode(symbols)
    assert decoded == data
