from libc.stdio cimport FILE
from cpython cimport (
  PyObject,
  PyBUF_READ,
  PyBUF_WRITE )

from cpython.memoryview cimport (
  PyMemoryView_FromMemory )

# cdef extern from "Python.h":
#   const int PyBUF_READ
#   const int PyBUF_WRITE
#   PyObject *PyMemoryView_FromMemory(char *mem, Py_ssize_t size, int flags)

import logging
log = logging.getLogger('p4est.core')

cimport numpy as npy
import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
sc_log_priority_to_level = [
  # SC_LP_ALWAYS
  logging.NOTSET,
  # SC_LP_TRACE
  logging.NOTSET + 5,
  # SC_LP_DEBUG
  logging.DEBUG,
  # SC_LP_VERBOSE
  logging.DEBUG + 5,
  # SC_LP_INFO
  logging.INFO,
  # SC_LP_STATISTICS
  logging.INFO + 1,
  # SC_LP_PRODUCTION
  logging.INFO + 2,
  # SC_LP_ESSENTIAL
  logging.INFO + 3,
  # SC_LP_ERROR
  logging.ERROR,
  # SC_LP_SILENT
  logging.CRITICAL + 10 ]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def log_initialize():
  sc_set_log_defaults(
    NULL,
    <sc_log_handler_t>_sc_log_handler,
    -1 )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef void _sc_log_handler(
  FILE * log_stream,
  const char *filename,
  int lineno,
  # NOTE: currently no API for getting 'package' information outside of libsc
  int package,
  int category,
  int priority,
  const char *msg) with gil:

  if msg == NULL:
    return

  level = sc_log_priority_to_level[min(9, max(0, priority))]

  if not log.isEnabledFor(level):
    return

  cdef bytes b_str

  if filename == NULL:
    filename_str = "(unknown file)"
  else:
    b_str = filename
    filename_str = b_str.decode('ascii', errors = 'replace')

  b_str = msg
  msg_str = b_str.decode('ascii', errors = 'replace')

  record = log.makeRecord(
    log.name,
    level,
    filename_str,
    lineno,
    msg_str,
    tuple(),
    None,
    "(unknown function)",
    None,
    None )

  log.handle(record)

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
cdef ndarray_from_sc_array(write, dtype, subitems, sc_array_t* arr):
  """

  .. attention::

    The returned array is simply a view into the underlying memory owned by the
    sc array, and should never be passed on to other code that cannot gaurantee
    the sc array will be kept alive.
    Attempting to access the array elements after the sc array is freed
    will (at best) lead to a segfault.

  Parameters
  ----------
  write : bool
    Should the returned ndarray be writable, otherwise is read-only
  dtype : np.dtype
    Numpy datatype of returned array.
  subitems : int
    Number of ``dtype`` items in the returned array per each item in the
    sc array.
    A minimal check will be performed to ensure that
    ``dtype.itemsize * subitems == arr.elem_size``.
  arr : sc_array_t*
    The sc array to create view of.

  Returns
  -------
  ndarray:
    Array of length ``arr->elem_count * subitems``.
  """

  itemsize = np.dtype(dtype).itemsize

  if itemsize * subitems != arr.elem_size:
    raise ValueError(f"Item size does not match: {dtype}.itemsize == {itemsize}, arr.elem_size == {arr.elem_size}")

  buffer = <object>PyMemoryView_FromMemory(
    arr.array,
    arr.elem_size * arr.elem_count,
    PyBUF_WRITE if write else PyBUF_READ )

  return np.frombuffer(
    buffer,
    dtype = dtype,
    count = subitems * arr.elem_count )
