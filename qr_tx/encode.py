from qr_tx.encoding import QREncoder
import cv2
from pathlib import Path

import argparse


def play_qr_data(p: Path, frame_rate: int, intial_delay: int = 2000):
    data = p.open("rb").read()
    encoder = QREncoder(data)
    cv2.namedWindow("data")
    delay = 1000 // frame_rate
    for idx, img in enumerate(encoder.qr_imgs()):
        cv2.imshow("data", img)
        if idx == 0:
            cv2.waitKey(intial_delay)
        else:
            cv2.waitKey(delay)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--frame_rate", type=int, default=5)
    args = parser.parse_args()

    play_qr_data(args.input_file, args.frame_rate)