# DNM
import argparse
import random
import string
import time

import h5py
import numpy as np
from h5py.to_arrow import read_vlen_dataset


def gen_data(N, K):
    choices = np.random.default_rng().choice(list(string.ascii_letters), 32 * K).reshape((-1, 32)).view("U32")[:, 0].astype("S32")
    return np.random.default_rng().choice(choices, N)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--read", choices=["object (bytes)", "object (str)", "npystrings", "pyarrow"])

    parser.add_argument("fname")
    args = parser.parse_args()

    if args.write:
        with h5py.File(args.fname, "w") as f:
            for N in range(1, 11):
                data = gen_data(N * 1_000_000, 100_000)
                f.create_dataset(
                     f"ascii/{N}M", 
                     dtype=h5py.string_dtype(encoding="ascii"), 
                     data=data,
                )
                f.create_dataset(
                     f"utf-8/{N}M", 
                     dtype=h5py.string_dtype(encoding="utf-8"), 
                     data=data,
                )                            

    else:
        with h5py.File(args.fname, "r") as f:
            for N in range(1, 11):
                t0 = time.time()
                if args.read == "object (str)":
                    dset = f[f"utf-8/{N}M"]
                    data = dset.asstr()[:]
                elif args.read == "object (bytes)":
                    dset = f[f"utf-8/{N}M"]
                    data = dset[:]
                elif args.read == "npystrings":
                    dset = f[f"utf-8/{N}M"]
                    data = dset[:]
                elif args.read == "pyarrow":
                    dset = f[f"ascii/{N}M"]
                    data = read_vlen_dataset(dset)
                t1 = time.time()
                assert len(data) == N * 1_000_000
                print(f"{args.read},{N}M,{t1 - t0}")


if __name__ == "__main__":
    main()
