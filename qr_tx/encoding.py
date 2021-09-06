from dataclasses import dataclass
import numpy as np
from typing import Iterable, Optional, Set, Tuple
import qrcode
from pyzbar import pyzbar
import io
import base64

FRAME_SZ = 2960 // 8
VERSION = 12
ECC_LVL = qrcode.constants.ERROR_CORRECT_L


@dataclass
class FrameData:
    seq: int
    total_frames: int
    data: bytes

    @staticmethod
    def uint_to_bytes(i) -> bytes:
        return base64.b64encode(
            i.to_bytes((i.bit_length() + 8) // 8, "big", signed=False)
        )

    @staticmethod
    def bytes_to_uint(b) -> int:
        return int.from_bytes(base64.b64decode(b), "big", signed=False)

    def encode(self) -> bytes:
        buffer = io.BytesIO()
        buffer.write(FrameData.uint_to_bytes(self.seq))
        buffer.write(FrameData.uint_to_bytes(self.total_frames))
        buffer.write(self.data)
        buffer.seek(0)
        return buffer.read()

    @classmethod
    def decode(cls, bin_data: bytes):
        seq = FrameData.bytes_to_uint(bin_data[:4])
        total_frames = FrameData.bytes_to_uint(bin_data[4:8])
        data = bin_data[8:]
        # seq, total_frames, data = msgpack.unpackb(bin_data, raw=True)
        return cls(seq, total_frames, data)

    @staticmethod
    def encode_to_frames(data: bytes) -> Iterable["FrameData"]:
        b64_data = base64.b64encode(data)
        data_chunk_sz = FRAME_SZ - 8
        chunks = [
            b64_data[i : i + data_chunk_sz]
            for i in range(0, len(b64_data), data_chunk_sz)
        ]
        n_frames = len(chunks)
        print(f"{len(data)} Bytes of data to be encoded to {n_frames} frames")
        for seq, chunk in enumerate(chunks):
            frame = FrameData(seq, n_frames, chunk)
            if seq == 0:
                print(f"frame sz: {len(frame.encode())}")
            yield frame


@dataclass
class QREncoder:
    data: bytes

    def qr_imgs(self) -> Iterable[np.array]:
        for frame in FrameData.encode_to_frames(self.data):
            yield QREncoder.create_qr_img(frame.encode())

    @staticmethod
    def create_qr_img(data: bytes) -> np.array:
        qr = qrcode.QRCode(
            version=VERSION,
            error_correction=ECC_LVL,
        )
        qr.add_data(data)
        img = qr.make_image().convert("RGB")
        return np.array(img)


@dataclass
class DecodeResult:
    data: Optional[bytes] = None
    missing_frames: Optional[Set[int]] = None


class QRDecoder:
    def __init__(self):
        self.frame_data = {}
        self.total_frames = 0
        self.n_frames_decoded = 0
        self.redundant_frames = 0
        self.seen_frames = 0

    def decode_qr_img(self, img: bytes):
        self.seen_frames += 1
        frame = QRDecoder.qr_img_to_frame(img)
        if frame is not None:
            if frame.seq not in self.frame_data:
                self.frame_data[frame.seq] = frame.data
                self.n_frames_decoded += 1
                self.total_frames = frame.total_frames
            else:
                self.redundant_frames += 1

    @staticmethod
    def qr_img_to_frame(img_data: np.array) -> Optional[FrameData]:
        decoded = pyzbar.decode(img_data, symbols=[pyzbar.ZBarSymbol.QRCODE])
        if len(decoded) == 1:
            raw_data = decoded[0].data
            return FrameData.decode(raw_data)

    def get_data(self) -> DecodeResult:
        if self.n_frames_decoded < self.total_frames:
            missing_frames = set(range(self.total_frames)) - set(self.frame_data.keys())
            return DecodeResult(missing_frames=missing_frames)
        buffer = io.BytesIO()
        for i in range(self.total_frames):
            buffer.write(self.frame_data[i])
        buffer.seek(0)
        b64_data = buffer.read()
        data = base64.b64decode(b64_data)
        print(f"{self.redundant_frames} / {self.seen_frames} frames are redundant")
        return DecodeResult(data=data)