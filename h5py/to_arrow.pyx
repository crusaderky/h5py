from libc.stdlib cimport free, malloc
from libc.string cimport strcpy, strlen
from libc.stdint cimport int32_t
import pyarrow
import sys

cdef extern from "hdf5.h":
    ctypedef int herr_t
    ctypedef long int hid_t
    cdef hid_t H5T_C_S1
    cdef size_t H5T_VARIABLE
    int H5S_ALL
    int H5P_DEFAULT

    hid_t H5Dget_space(hid_t dset_id)
    hid_t H5Tcopy(hid_t type_id)
    herr_t H5Tset_size(hid_t type_id, size_t size)
    herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t plist_id, void *buf) nogil
    herr_t H5Dvlen_reclaim(hid_t type_id, hid_t space_id,  hid_t plist, void *buf)
    herr_t H5Sclose(hid_t space_id)
    herr_t H5Tclose(hid_t type_id)


def read_vlen_dataset(dataset):
    cdef hid_t dset = dataset.id.id
    cdef hid_t space = H5Dget_space(dset)

    cdef hid_t memtype = H5Tcopy(H5T_C_S1)
    H5Tset_size(memtype, H5T_VARIABLE)

    cdef size_t nrows = len(dataset)
    cdef size_t i
    cdef char ** rdata = <char **> malloc(nrows * sizeof(char *))

    err = H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata)
    assert err >= 0
    offsets_buffer = pyarrow.allocate_buffer((nrows + 1) * sizeof(int32_t))

    cdef int32_t * offsets = <int32_t *> <long> offsets_buffer.address
    offsets[0] = 0
    for i in range(nrows):
        offsets[i + 1] = offsets[i] + strlen(rdata[i])
    data_buffer = pyarrow.allocate_buffer(offsets[nrows])
    cdef char * data = <char *> <long> data_buffer.address

    for i in range(nrows):
        strcpy(data + offsets[i], rdata[i])

    H5Dvlen_reclaim(memtype, space, H5P_DEFAULT, rdata)
    free(rdata)
    H5Sclose(space)
    H5Tclose(memtype)

    arr = pyarrow.Array.from_buffers(
        pyarrow.string(), nrows, [None, offsets_buffer, data_buffer],
    )
    return arr
