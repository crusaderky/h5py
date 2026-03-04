# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2026 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
Test multi-threaded concurrent access to the same Dataset.

Note that multi-threaded concurrent access to the same File, but on
different datasets, is extensively tested by the rest of the test suite
when it is executed with pytest-run-parallel. This is thanks to the fact
that all tests use a shared fixture / setUp that creates the File, which
is then shared by all the threads spawned by pytest-run-parallel.
"""

import numpy as np
import pytest
import threading

from .common import run_threaded

pytestmark = pytest.mark.thread_unsafe(reason="spawns thread pool itself")

N_WORKERS = 8
N_ROWS = 80  # divisible by N_WORKERS
N_COLS = 64
CHUNK_ROWS = N_ROWS // N_WORKERS  # 10 rows per chunk when separate_chunks


@pytest.mark.parametrize(
    "chunks",
    [
        # Each thread independently accesses a different chunk
        pytest.param((CHUNK_ROWS, N_COLS), id="independent"),
        # All threads share the same chunk
        pytest.param((N_ROWS, N_COLS), id="shared"),
    ],
)
def test_threaded_read(writable_file, chunks):
    """Multiple threads read from the same dataset in parallel."""
    data = np.arange(N_ROWS * N_COLS).reshape(N_ROWS, N_COLS)
    ds = writable_file.create_dataset("x", data=data, chunks=chunks)

    def reader(worker_id, barrier):
        barrier.wait()
        row_start = worker_id * CHUNK_ROWS
        row_end = row_start + CHUNK_ROWS
        expect = data[row_start:row_end, :]
        actual = ds[row_start:row_end, :]
        np.testing.assert_array_equal(actual, expect)

    run_threaded(
        reader,
        max_workers=N_WORKERS,
        pass_count=True,
        pass_barrier=True,
        outer_iterations=10,
    )


@pytest.mark.parametrize(
    "chunks",
    [
        # Each thread independently accesses a different chunk
        pytest.param((1, N_COLS), id="independent"),
        # All threads share the same chunk
        pytest.param((N_ROWS, N_COLS), id="shared"),
    ],
)
@pytest.mark.parametrize("iterations", [1, 10])
def test_threaded_read_write_different_rows(writable_file, chunks, iterations):
    """Multiple threads write and read on different areas of the same
    dataset at the same time.
    """
    data = np.arange(N_ROWS * N_COLS).reshape(N_ROWS, N_COLS)
    ds = writable_file.create_dataset(
        "x", shape=(N_ROWS, N_COLS), dtype=data.dtype, chunks=chunks
    )

    def work(worker_id, barrier):
        barrier.wait()
        idx = slice(worker_id, None, N_WORKERS)
        ds[idx] = data[idx]
        np.testing.assert_array_equal(ds[idx], data[idx])

    run_threaded(
        work,
        max_workers=N_WORKERS,
        pass_count=True,
        pass_barrier=True,
        outer_iterations=iterations,
    )

    np.testing.assert_array_equal(ds[:], data)


@pytest.mark.parametrize(
    "chunks",
    [
        # Read/write thread operates on chunks that are not being resized
        pytest.param((1, N_COLS), id="independent"),
        # Read/write thread operates on a chunk that is being resized
        pytest.param((N_ROWS, N_COLS), id="shared"),
    ],
)
def test_threaded_resize_grow(writable_file, chunks):
    """One thread enlarges the dataset by one row per iteration while
    another thread reads reads rows that are guaranteed to exist at the
    time it samples the dataset shape.
    """
    data = np.arange(N_ROWS * N_COLS).reshape(N_ROWS, N_COLS)
    ds = writable_file.create_dataset(
        "x",
        shape=(0, N_COLS),
        maxshape=(None, N_COLS),
        chunks=chunks,
        dtype=data.dtype,
    )
    ready = threading.Event()
    done = threading.Event()

    def grower():
        ds.resize(0, axis=0)
        done.clear()
        ready.set()
        for cur_rows in range(N_ROWS):
            ds.resize(cur_rows + 1, axis=0)
            ds[cur_rows] = data[cur_rows]
        done.set()
        ready.clear()

    def read_all():
        ready.wait()
        while not done.is_set():
            actual = ds[:]  # Never a race condition, shape-wise
            # There is a data race on the last row, which may or may not have been
            # already written by grower(). There are no race conditions on the previous
            # rows.
            expect = data[: actual.shape[0]]
            np.testing.assert_array_equal(actual[:-1], expect[:-1])
            # Note: the last row is guaranteed to be either all zeros or fully populated
            # because it's on a single chunk.
            try:
                np.testing.assert_array_equal(actual[-1], expect[-1])
            except AssertionError:
                np.testing.assert_array_equal(actual[-1], np.zeros_like(expect[-1]))

    def read_write_last():
        ready.wait()
        while not done.is_set():
            nrows = ds.shape[0]
            actual = ds[nrows - 1]  # Never a race condition
            expect = data[nrows - 1]
            try:
                np.testing.assert_array_equal(actual[-1], expect[-1])
            except AssertionError:
                np.testing.assert_array_equal(actual[-1], np.zeros_like(expect[-1]))
            ds[nrows - 1] = expect  # Never a race condition

    def work(worker_id, barrier):
        barrier.wait()
        if worker_id == 0:
            grower()
        elif worker_id % 2:
            read_all()
        else:
            read_write_last()

    run_threaded(
        work,
        max_workers=N_WORKERS,
        pass_count=True,
        pass_barrier=True,
        outer_iterations=10,
    )
    np.testing.assert_array_equal(ds[:], data)


@pytest.mark.parametrize(
    "chunks",
    [
        # Entire chunks are deleted at once
        pytest.param((1, N_COLS), id="independent"),
        # Chunks are partially shrunk and updated
        pytest.param((N_ROWS, N_COLS), id="shared"),
    ],
)
def test_threaded_resize_shrink(writable_file, chunks):
    """One thread shrinks the dataset by one row per iteration while
    other threads read and write on it.  Because the dataset is being shrunk
    concurrently, I/O on the reader/writer threads is prone to race conditions where the
    requested surface may not exist anymore. Test that they are handled gracefully with
    an IndexError.
    """
    data = np.arange(N_ROWS * N_COLS).reshape(N_ROWS, N_COLS)
    ds = writable_file.create_dataset(
        "x",
        shape=(0, N_COLS),
        maxshape=(None, N_COLS),
        chunks=chunks,
        dtype=data.dtype,
    )
    ready = threading.Event()

    def shrinker():
        ds.resize(N_ROWS, axis=0)
        ds[:] = data
        ready.set()
        while ds.shape[0]:
            ds.resize(ds.shape[0] - 1, axis=0)
        ready.clear()

    def read_all():
        ready.wait()
        while True:
            actual = ds[:]  # Never a race condition
            if not actual.shape[0]:
                break
            expect = data[: actual.shape[0]]
            np.testing.assert_array_equal(actual, expect)

    def read_write_last():
        ready.wait()
        while True:
            nrows = ds.shape[0]
            expect = data[nrows - 1]

            if not nrows:
                break
            try:
                # Race conditions - dataset may already be smaller
                actual = ds[nrows - 1]
                ds[nrows - 1] = expect
            except IndexError:
                continue  # Handled gracefully

            np.testing.assert_array_equal(actual, expect)

    def worker(worker_id, barrier):
        barrier.wait()
        if worker_id == 0:
            shrinker()
        elif worker_id % 2:
            read_all()
        else:
            read_write_last()

    run_threaded(
        worker,
        max_workers=N_WORKERS,
        pass_count=True,
        pass_barrier=True,
        outer_iterations=10,
    )
    assert ds.shape == (0, N_COLS)
