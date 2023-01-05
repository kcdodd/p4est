from cpython cimport (
  PyObject,
  Py_INCREF,
  PyBUF_READ,
  PyBUF_WRITE )

# from cpython.buffer cimport (
#   PyObject_GetBuffer,
#   PyBuffer_Release,
#   PyBUF_ANY_CONTIGUOUS,
#   PyBUF_SIMPLE )

from cpython.memoryview cimport (
  PyMemoryView_FromMemory )

cimport numpy as npy
npy.import_array()
import numpy as np

from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype

import sys

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef ndarray_from_ptr(write, dtype, count, char* arr):
  """

  .. attention::

    The returned array is simply a view into the underlying memory pointed to,
    and should never be passed on to other code that cannot gaurantee
    the referenced memory will be kept alive.
    Attempting to access the array elements after the pointer is freed
    will (at best) lead to a segfault.

    There is also no way for this routine to verify
    the amount of referenced memory matches the requested item count.

  Parameters
  ----------
  write : bool
    Should the returned ndarray be writable, otherwise is read-only
  dtype : np.dtype
    Numpy datatype of returned array.
  count : int
    Number of items of dtype.
  arr : char*
    Pointer to beginning of memory, must be at least ``dtype.itemsize * count``.

  Returns
  -------
  ndarray:
    Array of length ``count``.
  """

  itemsize = np.dtype(dtype).itemsize

  buffer = <object>PyMemoryView_FromMemory(
    arr,
    itemsize * count,
    PyBUF_WRITE if write else PyBUF_READ )

  return np.frombuffer(
    buffer,
    dtype = dtype,
    count = count )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ndarray_bufspec(arr):
  return _ndarray_bufspec(arr)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef _ndarray_bufspec(arr):
  """Create an MPI Datatype for an ndarray, potentially sliced
  """
  mpi_dtype = MPI.Datatype(from_numpy_dtype(arr.dtype))

  itemsize = arr.itemsize

  # check which order is more likely to result in contiguous blocks
  if arr.strides[0] < arr.strides[-1]:
    # Fortran style, first is fastest
    dims = range(arr.ndim)

  else:
    # c-array style, last is fastest
    dims = range(arr.ndim-1, -1, -1)

  contiguous_vol = 1

  for dim in dims:

    item_stride = arr.strides[dim] // itemsize
    # print(f"dim {dim}: contiguous_vol= {contiguous_vol}, count= {arr.shape[dim]}, stride= {item_stride}")

    if contiguous_vol > 0 and ( arr.shape[dim] == 1 or item_stride == contiguous_vol ):
      # if possible, collapse dimensions together into single contiguous block
      # * arr.shape[dim] == 1: This dimension isn't iterated over, so if its
      #   elements are already contiguous, then this won't break it
      # * item_stride == contiguous_vol: If the stride for this dimension is
      #   equal to its effective element size, then there are no gaps
      contiguous_vol *= arr.shape[dim]

    else:
      # number of elements of the *current* Datatype grouped into each block.
      # If multiple dimensions were initially collapsed into a single contiguous
      # block, that is used.
      # Otherwise the count is 1, since the current Datatype already encompases
      # all the 'sub-blocks'
      count = contiguous_vol or 1

      # once non-contiguous, it can never become contiguous by adding dimensions
      contiguous_vol = 0

      # NOTE: for some reason, the 'Create_vector' variant does not seem to
      # create the correct offsets, or otherwise mangles the data.
      mpi_dtype = mpi_dtype.Create_hvector(
        # Number of blocks
        arr.shape[dim],
        # Number of elements in each block
        count,
        # Number of bytes between start of each block
        arr.strides[dim] )

  # the number of blocks of the final Datatype.
  # If contiguous_vol > 0, then this will simply be the total number of items
  # of the initial dtype.
  count = contiguous_vol or 1

  mpi_dtype.Commit()

  # NOTE: this is some hackery to create a buffer that mpi4py will accept
  # (since non-contiguous buffers are automatically rejected).
  # In the case that 'arr' is non-contiguous, there is some difficulty to
  # retreiving the actual memory footprint from here, and various ways of
  # attempting to get a pointer to the end of 'arr' seem pretty fragil.
  # This method first gets back to the base (contiguous) array, then uses its
  # size to find a 'safe' end to the memory.
  # As long as the 'start' pointer and datatype strides lead to valid memory,
  # MPI doesn't actually care about these other numbers.

  base = arr

  while base.base is not None:
    base = base.base

  cdef npy.ndarray _base = base
  cdef npy.ndarray _arr = arr

  cdef ptrdiff_t nbytes = base.nbytes
  cdef npy.npy_int8* origin = <npy.npy_int8*>_base.data
  cdef npy.npy_int8* start = <npy.npy_int8*>_arr.data
  cdef npy.npy_int8* end = origin + nbytes

  nbytes = end - start
  n = nbytes // itemsize

  cdef npy.npy_intp[:] _dims = np.array([nbytes], dtype = np.intp)

  cdef npy.ndarray buf = npy.PyArray_New(
    npy.ndarray,
    1,
    &_dims[0],
    npy.NPY_INT8,
    NULL,
    <void*>start,
    0,
    npy.NPY_ARRAY_C_CONTIGUOUS | (npy.NPY_ARRAY_WRITEABLE if arr.flags.writeable else 0),
    None )

  # Reference count of the 'contiguous' buffer to the original.
  # npy.set_array_base(buf, base)
  assert npy.PyArray_SetBaseObject(buf, arr) == 0
  Py_INCREF(arr)

  return [
    buf,
    count,
    mpi_dtype]