import argparse
from pathlib import Path
from typing import Optional

import cv2

from qr_tx.encoding import QRDecoder


def decode_qr_data(p: Path):

    vidcap = cv2.VideoCapture(str(p))
    decoder = QRDecoder()
    count = 0
    while vidcap.isOpened():
        success, img = vidcap.read()
        if success:
            decoder.decode_qr_img(img)
            if count % 100 == 0:
                print(f"Frame: {count}")
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
    return decoder.get_data()


def handle_result(res: Optional[bytes], out: Path):
    if res is None:
        print(f"Unable to decode")
    else:
        with out.open("wb") as f:
            f.write(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    res = decode_qr_data(args.input_file)
    handle_result(res, args.output_file)
