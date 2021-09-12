import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2

from qr_tx.encoding import QRDecoder
from qr_tx.util import setup_logging

log = logging.getLogger(__name__)


def decode_qr_data(p: Path):
    """Open input video and decode data"""
    vidcap = cv2.VideoCapture(str(p))
    decoder = QRDecoder()
    count = 0
    while vidcap.isOpened():
        success, img = vidcap.read()
        if success:
            decoder.decode_qr_img(img)
            if count % 100 == 0:
                log.debug(f"Raw Video Frame: {count}")
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
    return decoder.get_data()


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    res = decode_qr_data(args.input_file)
    if res is None:
        raise ValueError("Unable to decode - not enough data received")
    else:
        with args.output_file.open("wb") as f:
            f.write(res)
