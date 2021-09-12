# qr_tx

Proof-of-concept library to transmit data as a stream of QR codes.
![Demo](https://media.giphy.com/media/ypmC7vwdqE6X83hQRz/giphy.gif)

## Encoding

Data is loaded from an input file as raw binary and first encoded using [Luby Transform](https://en.wikipedia.org/wiki/Luby_transform_code) code, a class of erasure codes, to
add redudancy and facilitate lossy transmission. The LT encoded output symbols are then encoded to a stream of QR codes that are displayed on screen to be recorded as the transmission.

## Decoding

An input video file is parsed frame-by-frame to extract the LT encoded symbol data from detected QR codes. After all frames are parsed, the resulting set of symbols is decoded to the orignal message if possible.

## Commandline Usage

Create python env (conda):
```
conda env create -f environment.yml
conda activate qr_tx
```

Encoding:
```bash
python -m qr_tx.encode <file>
```

Decoding:
```bash
python -m qr_tx.decode <movie_file> <output_file>
```

## Potential Improvements
Integrity:
 - Use a PRNG in the LT encoding implementation to avoid issues with numpy version compatability (current implementation depends on seed values)

Performance:
- Use a more performant QR encoding library
- Parallelize the QR frame encoding process


Tuning:
 - Try different QR code versions (sizes). Current size chosen partly because larger capacity QR versions take too long to encode with current QR library
 - Benchmark optimal redundancy level based on expected symbol loss ([code rate](https://en.wikipedia.org/wiki/Code_rate)). For example, current implementation drops only around 1% of symbols when using screen recording,
    yet 2X redundancy can handle well over 10% symbol loss. What is expected loss when videoing a screen with a phone or other device?
