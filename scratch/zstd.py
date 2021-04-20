import zstandard
import numpy as np

cctx = zstandard.ZstdCompressor(level=22)
dctx = zstandard.ZstdDecompressor()

with open("/tmp/zstd_comp.txt", "wb") as f:
    compressor = cctx.stream_writer(f)
    compressor.write(np.zeros(300).astype(np.uint8))
    compressor.write(np.ones(300).astype(np.float32))
    print(compressor.flush())

with open("/tmp/zstd_comp.txt", "rb") as f:
    with dctx.stream_reader(f) as reader:
        dec = reader.read()
        y = np.frombuffer(dec, dtype=np.uint8, count=300)
        z = np.frombuffer(dec, dtype=np.float32, count=300, offset=300)
        # dec = reader.read(size=300 * 8)
        # z = np.frombuffer(dec, dtype=np.float64)
