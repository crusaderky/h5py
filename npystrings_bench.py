# DNM
import argparse
import os
import random
import string
import time

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-width", type=int, default=0)
    parser.add_argument("--max-width", type=int, default=100)
    parser.add_argument("--num-strings", type=int, default=1_000_000)
    parser.add_argument("--dtype", type=str, choices=["str", "bytes", "T"], required=True)
    args = parser.parse_args()

    data = [
        "".join(
            random.choices(
                string.ascii_letters,
                k=random.randint(args.min_width, args.max_width),
            )
        )
        for _ in range(args.num_strings)
    ]
    data_dtype = "O" if args.dtype in ("str", "bytes") else "T"
    dset_dtype = (
        h5py.string_dtype(encoding="utf-8" if args.dtype == "str" else "ascii")
        if args.dtype in ("str", "bytes") else "T"
    )
    data = np.asarray(data, dtype=data_dtype)
    print(f"input dtype: {data.dtype}")

    with h5py.File("bench.h5", "w") as f:
        t0 = time.time()
        ds = f.create_dataset("data", data=data, dtype=dset_dtype)
        t1 = time.time()
        print(f"dset dtype: {ds.dtype} metadata={ds.dtype.metadata}")

    print(f"write: {t1 - t0:.3f}s")
    with h5py.File("bench.h5", "r") as f:
        t0 = time.time()
        ds = f["data"]
        if args.dtype == "T":
            ds = ds.astype("T")
        elif args.dtype == "str":
            ds = ds.asstr()
        print(f"dset dtype: {ds.dtype} metadata={ds.dtype.metadata}")
        data2 = ds[:]
        t1 = time.time()
    print(f"output dtype: {data2.dtype}")
    print(f"read: {t1 - t0:.3f}s")

    if args.dtype == "bytes":
        data2 = np.asarray([d.decode("utf-8") for d in data2])

    np.testing.assert_array_equal(data2, data)
    os.remove("bench.h5")


if __name__ == "__main__":
    main()
