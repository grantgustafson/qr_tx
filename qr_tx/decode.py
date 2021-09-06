from qr_tx.encoding import QRDecoder, DecodeResult
import cv2
import base64
from pathlib import Path

import argparse


def decode_qr_data(p: Path):

    vidcap = cv2.VideoCapture(str(p))
    decoder = QRDecoder()
    count = 0
    while vidcap.isOpened():
        success, img = vidcap.read()
        if success:
            decoder.decode_qr_img(img)
            if count % 10 == 0 and False:
                cv2.imshow("test", img)
                cv2.waitKey(200)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
    return decoder.get_data()


def handle_result(res: DecodeResult, out: Path):
    if res.missing_frames is not None:
        print(f"Missing Frames:\n{res.missing_frames}")
    else:
        with out.open("wb") as f:
            f.write(res.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    res = decode_qr_data(args.input_file)
    handle_result(res, args.output_file)