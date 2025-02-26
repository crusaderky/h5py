# DNM
import os
import h5py

with h5py.File("foo.h5", "w") as f:
    data = ["foo", "Hello world this is a very long string indeed", "bar"]

    print("write obj")
    ds = f.create_dataset("obj", data=data, dtype=h5py.string_dtype())
    print(ds)
    a = ds[:]
    print(repr(a))
    print("=" * 80)

    print("write native")
    ds = f.create_dataset("native", data=data, dtype="T")
    print(ds)
    a = ds[:]
    print(repr(a))
    print("=" * 80)

with h5py.File("foo.h5", "r") as f:
    print("read obj")
    ds = f["obj"]
    print(ds)
    a = ds[:]
    print(repr(a))
    print("=" * 80)

    print("read native")
    ds = f["native"]
    print(ds)
    print("astype('T')")
    ds = ds.astype("T")
    print(ds)
    a = ds[:]
    print(repr(a))
    print("rolling back")
    ds = ds.asstr()
    print(ds)
    print(ds._dset)
    a = ds[:]
    print(repr(a))
    print("=" * 80)

os.remove("foo.h5")
