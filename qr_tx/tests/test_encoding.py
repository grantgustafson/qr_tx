import random
import string

from qr_tx.encoding import QRDecoder, QREncoder, QRFrameData


def test_frame_data():
    data = "some test data"
    ser_frame = QRFrameData(1, 2, 3, 4, data).encode()
    deser_frame = QRFrameData.decode(ser_frame)
    assert deser_frame.idx == 1
    assert deser_frame.degree == 2
    assert deser_frame.n_blocks == 3
    assert deser_frame.n_total_bytes == 4
    assert deser_frame.data == data


def test_encoder():
    data = "".join(random.choice(string.ascii_letters) for _ in range(1000)).encode(
        "utf-8"
    )
    encoder = QREncoder(data)
    for frame in encoder.encode_qr_stream():
        assert len(frame) > 100


def test_single_frame_encode_decode():
    data = "SOME TEST DATA"
    frame_data = QRFrameData(1, 2, 3, 4, data).encode()
    img_data = QREncoder.create_qr_img(frame_data)
    frame = QRDecoder.qr_img_to_frame(img_data)
    assert frame is not None
    assert frame.idx == 1
    assert frame.degree == 2
    assert frame.n_blocks == 3
    assert frame.n_total_bytes == 4
    assert frame.data == data


def test_qr_encoding_decoding():
    data = "".join(random.choice(string.ascii_letters) for _ in range(8000)).encode(
        "utf-8"
    )
    encoder = QREncoder(data)
    decoder = QRDecoder()
    for i, frame in enumerate(encoder.encode_qr_stream()):
        decoder.decode_qr_img(frame)
    res = decoder.get_data()
    assert res is not None
    assert res == data
