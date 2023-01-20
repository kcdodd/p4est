# Enable postponed evaluation of annotations
from __future__ import annotations
try:
  from typing import (
    Optional,
    Union,
    Literal,
    TypeVar,
    NewType )
  from .typing import N, M
except:
  pass

from copy import copy
from collections.abc import Sequence
import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class jagged_array(Sequence):
  """Container for jagged (ragged) array

  Eeach 'row' potentially has a different number of entries, but also each row
  is contiguous (not a general 'sparse' array).
  The array is stored as a single contiguous array of one less dimension.
  Similar to compressed sparse row format, except that 'column' indices are not
  explicitly stored.

  Parameters
  ----------
  data :
    The data of all rows
  row_idx :
    Indices of each row, where row ``i`` is ``data[row_idx[i]:row_idx[i+1]]``,
    with ``row_idx[0] == 0`` and ``row_idx[-1] == len(data)``.

  """
  #-----------------------------------------------------------------------------
  def __init__(self,
    data : np.ndarray[(N, ...)],
    row_idx : np.ndarray[(M,), np.dtype[np.integer]]):

    row_idx = np.ascontiguousarray(row_idx)

    if not np.issubdtype(row_idx.dtype, np.integer):
      raise ValueError(f"Must have integral row_idx.dtype: {row_idx.dtype}")

    if (
      row_idx.ndim != 1
      or len(row_idx) == 0 ):

      raise ValueError(f"Must have row_idx.ndim = 1 and len(row_idx) > 0: {row_idx.ndim}, {len(row_idx)}")

    if row_idx[0] != 0:
      raise ValueError(f"Must have row_idx[0] = 0: {row_idx[0]}")

    if row_idx[-1] != len(data):
      raise ValueError(f"Must have row_idx[-1] = len(data) = {len(data)}: {row_idx[-1]}")

    row_counts = np.diff(row_idx)

    if not np.all(row_counts >= 0):
      raise ValueError(f"Must have row_idx[i] <= row_idx[i+i]")


    self._data = data
    self._row_idx = row_idx
    self._row_counts = row_counts

  #-----------------------------------------------------------------------------
  def __class_getitem__(cls, *args):
    from types import GenericAlias
    return GenericAlias(cls, args)

  #-----------------------------------------------------------------------------
  def __copy__(self):
    cls = type(self)
    arr = cls.__new__(cls)

    arr._data = copy(self._data)
    arr._row_idx = copy(self._row_idx)
    arr._row_counts = copy(self._row_counts)

    return arr

  #-----------------------------------------------------------------------------
  @property
  def flat(self) -> : np.ndarray[(N, ...)]:
    return self._data

  #-----------------------------------------------------------------------------
  @property
  def row_idx(self) -> np.ndarray[(M,), np.dtype[np.integer]]:
    return self._row_idx

  #-----------------------------------------------------------------------------
  @property
  def row_counts(self):
    return self._row_counts

  #-----------------------------------------------------------------------------
  @property
  def dtype(self) -> np.dtype:
    return self._data.dtype

  #-----------------------------------------------------------------------------
  def __len__(self):
    return len(self.row_idx) - 1

  #-----------------------------------------------------------------------------
  def __iter__(self):
    for i in range(len(self.row_idx) - 1):
      yield self[i]

  #-----------------------------------------------------------------------------
  def __getitem__( self, idx ):

    if isinstance(idx, int):
      return self._data[self.row_idx[idx]:self.row_idx[idx+1]]

    mask = np.zeros(len(self), dtype = bool)
    mask[idx] = True

    data_mask = np.repeat( mask, self._row_counts )

    return type(self)(
      data = self._data[data_mask],
      row_idx = np.concatenate(([0],np.cumsum(self._row_counts[mask]))).astype(np.int32) )

  #-----------------------------------------------------------------------------
  def __setitem__( self, idx, row_data ):
    self._data[self.row_idx[idx]:self.row_idx[idx+1]] = row_data

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def unique_full(
  arr : np.ndarray[(...,N,...), np.dtype[np.number]],
  axis : int ) \
  -> tuple[
    np.ndarray[(N,), np.dtype[np.integer]],
    np.ndarray[(N,), np.dtype[np.bool]],
    np.ndarray[(M,), np.dtype[np.integer]],
    np.ndarray[(N,), np.dtype[np.integer]] ]:
  r"""This is a reimplementation of np.unique, but returns the needed intermediate
  arrays needed to construct the outputs, instead of the standard outputs
  (numpy/lib/arraysetops.py).

  Parameters
  ----------
  arr :
    Values to sort along given axis.
  axis :
    Axis to perform sort.

  Returns
  -------
  sort_idx :
    The 'argsort' of the array along given axis, with all other axes collapsed
    to participate in sorting.
    ``unique_with_repeats = arr[sort_idx]``
  unique_mask :
    Mask of the first occurance of each unique value in the *sorted*
    array along the given axis.
    ``unique = arr[sort_idx[unique_mask]]``
  unique_idx :
    Offset to the first occurance of each unique value in the *sorted*
    array along the given axis, plus an additional entry at the end that
    is the total count.
    ``unique = arr[sort_idx[unique_idx[:-1]]]``
    ``counts = np.diff(unique_idx)``
  inv_idx :
    Indices of sorted unique values to reconstruct the original (unsorted) array.
    ``np.all(arr == unique[inv_idx])``
  """

  arr = np.moveaxis(arr, axis, 0)

  orig_shape, orig_dtype = arr.shape, arr.dtype
  n = orig_shape[0]
  m = np.prod(orig_shape[1:], dtype=np.intp)

  arr = np.ascontiguousarray(arr.reshape(n,m))

  dtype = [('f{i}'.format(i=i), arr.dtype) for i in range(arr.shape[1])]
  arr = arr.view(dtype).reshape(n)

  sort_idx = np.argsort(arr, kind='mergesort')

  arr = arr[sort_idx]

  # mask of first occurance of each value
  unique_mask = np.empty(n, dtype = bool)
  unique_mask[0] = True
  unique_mask[1:] = arr[1:] != arr[:-1]

  # get the indices of the first occurence of each value
  # NOTE: 0 < len(unique_idx) <= len(arr[axis])+1
  # NOTE: 'nonzero' returns a tuple of arrays
  unique_idx = np.concatenate(np.nonzero(unique_mask) + ([n],)).astype(np.intp)

  inv_idx = np.empty(unique_mask.shape, dtype = np.intp)
  inv_idx[sort_idx] = np.cumsum(unique_mask) - 1

  return sort_idx, unique_mask, unique_idx, inv_idx