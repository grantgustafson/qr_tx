import argparse
import logging
import time
from pathlib import Path

import cv2

from qr_tx.encoding import QREncoder
from qr_tx.util import setup_logging

log = logging.getLogger(__name__)


def display_qr_data(p: Path, frame_rate: int, redundancy: float):
    """
    Read data, encode to QR images, and display on screen.
    """
    data = p.open("rb").read()
    encoder = QREncoder(data)
    cv2.namedWindow("data")

    # delay based on desired frame rate - in practice frame processing time much longer
    # display loop strategy:
    # 1) display current frame
    # 2) compute next frame
    # 3) delay if any longer delay required based on frame rate
    delay = 1000 // frame_rate
    frame_iter = encoder.encode_qr_stream(redundancy)
    frame = next(frame_iter)
    frame_start_t = time.time()
    for idx, next_frame in enumerate(frame_iter):
        processing_time = round((time.time() - frame_start_t) * 1000)
        if idx % 100 == 0:
            log.debug(f"Frame {idx} processing time ms: {processing_time}")
        cv2.waitKey(max(delay - processing_time, 1))
        cv2.imshow("data", frame)
        frame = next_frame
        frame_start_t = time.time()
    cv2.imshow("data", frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--frame_rate", type=int, default=10)
    parser.add_argument("--redundancy", type=float, required=False, default=2.0)
    args = parser.parse_args()

    display_qr_data(args.input_file, args.frame_rate, args.redundancy)
