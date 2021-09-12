import base64
import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Set, Tuple

import crc32c
import numpy as np
import qrcode
from pyzbar import pyzbar

from qr_tx import lt

log = logging.getLogger(__name__)

# config about QR image data capacity
FRAME_SZ = 1046
VERSION = 18
ECC_LVL = qrcode.constants.ERROR_CORRECT_L
HEADER_SZ = 48


def get_block_sz() -> int:
    """
    Naive way to determine how much encoded data we can store in one QR image
    Will always return of multiple of 8 since is required by LT encoding implementation.
    """
    sz = FRAME_SZ // 2
    while True:
        b = b"b" * sz
        encoded_sz = len(base64.b32encode(b))
        if encoded_sz > FRAME_SZ:
            assert len(base64.b32encode(b"b" * (sz - 1))) <= FRAME_SZ
            sz -= 1
            return sz - (sz % 8)
        sz += 1


def bytes_to_alphanumneric(b: bytes) -> str:
    """Encode raw bytes to QR-friendly alphanumeric"""
    return base64.b32encode(b).decode("ascii").replace("=", "%")


def alphanumeric_to_bytes(d: str) -> bytes:
    """Decode QR alphanumeric back to bytes"""
    return base64.b32decode(d.replace("%", "="))


@dataclass
class QRFrameData:
    """
    Container for data encoded into a QR image from LT Symbol data.
    Data is represented as QR alphanumeric (see https://en.wikipedia.org/wiki/QR_code#Encoding).
    Note: QR supports "binary" in the form of ISO 8859-1 which isn't as convenient
    """

    idx: int
    degree: int
    n_blocks: int
    n_total_bytes: int
    data: str

    @staticmethod
    def pack_int(
        a: int,
        b: int,
        c: int,
    ) -> int:
        return a << 32 | b << 16 | c

    @staticmethod
    def unpack_int(x) -> Tuple[int, int, int]:
        c = x & ((1 << 16) - 1)
        b = x >> 16 & ((1 << 16) - 1)
        a = x >> 32 & ((1 << 16) - 1)
        return a, b, c

    @staticmethod
    def uint_to_alphanum(i: int) -> bytes:
        return bytes_to_alphanumneric(i.to_bytes(8, "big", signed=False))

    @staticmethod
    def alphanum_to_uint(b: bytes) -> int:
        return int.from_bytes(alphanumeric_to_bytes(b), "big", signed=False)

    def encode(self) -> str:
        n_total_bytes = QRFrameData.uint_to_alphanum(self.n_total_bytes)
        header = (
            QRFrameData.uint_to_alphanum(
                QRFrameData.pack_int(self.idx, self.degree, self.n_blocks)
            )
            + n_total_bytes
        )
        assert len(header) == 32
        data = "".join([header, self.data])
        checksum = crc32c.crc32c(data.encode("ascii"))
        checksum = QRFrameData.uint_to_alphanum(checksum)
        assert len(checksum) == 16
        return checksum + data

    @classmethod
    def decode(cls, alphanum_data: str):
        expected_checksum = QRFrameData.alphanum_to_uint(alphanum_data[:16])
        checksum = crc32c.crc32c(alphanum_data[16:].encode("ascii"))
        if checksum != expected_checksum:
            return None
        idx, degree, n_blocks = QRFrameData.unpack_int(
            QRFrameData.alphanum_to_uint(alphanum_data[16:32])
        )
        n_total_bytes = QRFrameData.alphanum_to_uint(alphanum_data[32:HEADER_SZ])
        data = alphanum_data[HEADER_SZ:]
        return cls(idx, degree, n_blocks, n_total_bytes, data)


@dataclass
class QREncoder:
    """Class to handle encoding raw data to stream of encoded QR images"""

    data: bytes

    def encode_qr_stream(self, redundancy: float = 2.0) -> Iterable[np.array]:
        """Encode raw data to QR image stream"""
        block_sz = get_block_sz() - HEADER_SZ
        symbols = lt.encode(self.data, block_sz, redundancy)
        log.info(f"Encoded {len(self.data)} bytes to {len(symbols)} frames")
        for symbol in symbols:
            frame = QRFrameData(
                symbol.idx,
                symbol.degree,
                symbol.n_blocks,
                symbol.n_total_bytes,
                bytes_to_alphanumneric(symbol.data_as_bytes()),
            )
            yield QREncoder.create_qr_img(frame.encode())

    @staticmethod
    def create_qr_img(data: str) -> np.array:
        """Encode alphanumeric data to a QR image"""
        qr = qrcode.QRCode(
            version=VERSION,
            error_correction=ECC_LVL,
        )
        qr.add_data(qrcode.util.QRData(data, mode=qrcode.util.MODE_ALPHA_NUM))
        img = qr.make_image().convert("RGB")
        return np.array(img)


class QRDecoder:
    """Class to organize decoding tooling"""

    def __init__(self):
        self.frame_data = {}
        self.n_frames_decoded = 0
        self.redundant_frames = 0
        self.seen_frames = 0

    def decode_qr_img(self, img: bytes):
        """Decode a single QR image of data and store results if not seen before"""
        self.seen_frames += 1
        frame = QRDecoder.qr_img_to_frame(img)
        if frame is not None:
            if frame.idx not in self.frame_data:
                self.frame_data[frame.idx] = frame
                self.n_frames_decoded += 1
                if frame.idx % 100 == 0:
                    log.debug(f"Symbol idx: {frame.idx}")
            else:
                self.redundant_frames += 1

    @staticmethod
    def qr_img_to_frame(img_data: np.array) -> Optional[QRFrameData]:
        """Extract frame data from input image if possible"""
        decoded = pyzbar.decode(img_data, symbols=[pyzbar.ZBarSymbol.QRCODE])
        if len(decoded) == 1:
            raw_data = decoded[0].data.decode("ascii")
            return QRFrameData.decode(raw_data)

    def get_data(self) -> Optional[bytes]:
        """
        Attempt to extract message from seen frames of data.
        Will fail if not enough data seen to successfully decode.
        """
        symbols = [
            lt.Symbol(
                f.idx,
                f.degree,
                f.n_blocks,
                f.n_total_bytes,
                lt.Symbol.data_from_bytes(alphanumeric_to_bytes(f.data)),
            )
            for f in self.frame_data.values()
        ]

        data = lt.decode(symbols)
        log.debug(f"{self.redundant_frames} / {self.seen_frames} frames are redundant")
        return data
