import argparse
import time
from pathlib import Path

import cv2

from qr_tx.encoding import QREncoder


def play_qr_data(p: Path, frame_rate: int, intial_delay: int = 2000):
    data = p.open("rb").read()
    encoder = QREncoder(data)
    cv2.namedWindow("data")
    delay = 1000 // frame_rate
    frame_iter = encoder.encode_qr_stream()
    frame = next(frame_iter)
    frame_start_t = 0
    for idx, next_frame in enumerate(frame_iter):
        if idx == 1:
            cv2.waitKey(intial_delay)
        else:
            processing_time = (time.time() - frame_start_t) * 1000
            if idx % 100 == 0:
                print(f"Frame {idx} processing time ms: {processing_time}")
            cv2.waitKey(max(delay - processing_time, 1))
        cv2.imshow("data", frame)
        frame = next_frame
        frame_start_t = time.time()
    cv2.imshow("data", frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--frame_rate", type=int, default=5)
    args = parser.parse_args()

    play_qr_data(args.input_file, args.frame_rate)
