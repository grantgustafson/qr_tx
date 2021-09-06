from qr_exfil.encoding import FrameData, QRDecoder, QREncoder
import random
import string
from PIL import Image


def test_frame_data():
    data = "some test data".encode("utf-8")
    ser_frame = FrameData(1, 1, data).encode()
    print(ser_frame)
    deser_frame = FrameData.decode(ser_frame)
    assert deser_frame.seq == 1
    assert deser_frame.total_frames == 1
    assert deser_frame.data == data


def test_encoder():
    data = "".join(random.choice(string.ascii_letters) for _ in range(2000)).encode(
        "utf-8"
    )
    encoder = QREncoder(data)
    for frame in encoder.qr_imgs():
        assert len(frame) > 100


def test_single_frame():
    data = "some test data".encode("utf-8")
    frame_data = FrameData(1, 2, data).encode()
    img_data = QREncoder.create_qr_img(frame_data)
    frame = QRDecoder.qr_img_to_frame(img_data)
    assert frame is not None
    assert frame.seq == 1
    assert frame.total_frames == 2
    assert frame.data == data


def test_qr_encoding_decoding():
    data = "".join(random.choice(string.ascii_letters) for _ in range(2000)).encode(
        "utf-8"
    )
    encoder = QREncoder(data)
    decoder = QRDecoder()
    for i, frame in enumerate(encoder.qr_imgs()):
        print(f"Frame: {i}")
        decoder.decode_qr_img(frame)
    print(decoder.get_data())
