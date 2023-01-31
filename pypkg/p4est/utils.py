# Enable postponed evaluation of annotations
from __future__ import annotations
from partis.utils import TYPING

if TYPING:
  from typing import (
    Union,
    Literal)
  from .typing import N, M

from abc import ABCMeta
from collections import namedtuple
from copy import copy
from collections.abc import (
  Sequence,
  Mapping)
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
  def flat(self) -> np.ndarray[(N, ...)]:
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
    np.ndarray[(N,), np.dtype[bool]],
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
    Axis to perform sort, ``arr.shape[axis] == N``


  Returns
  -------
  sort_idx :
    The 'argsort' of the array along given axis, with all other axes collapsed
    to participate in sorting.

  unique_mask :
    Mask of the first occurrence of each unique value in the *sorted*
    array along the given axis.

  unique_idx :
    Offset to the first occurrence of each unique value in the *sorted*
    array along the given axis, plus an additional entry at the end that
    is the total count.

  inv_idx :
    Indices of sorted unique values to reconstruct the original (unsorted) array.



  Examples
  --------

  .. code-block:: python

    sort_idx, unique_mask, unique_idx, inv_idx = unique_full(arr, axis = 0)

    unique = arr[sort_idx[unique_mask]]
    unique = arr[sort_idx[unique_idx[:-1]]]
    np.all(arr == unique[inv_idx])

    unique_with_repeats = arr[sort_idx]
    counts = np.diff(unique_idx)
    np.all(unique_with_repeats == np.repeat(unique, counts))

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


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class InfoField:
  """Descriptor for accessing info fields
  """
  __slots__ = ('_shape', '_dtype', '_name', '_idx', 'fget')

  #-----------------------------------------------------------------------------
  def __init__( self, shape, dtype, *args, **kwargs ):

    self._shape = shape
    self._dtype = dtype
    self._name = None
    self._idx = None

    def dummy_fget():
      pass

    if TYPING:
      # NOTE: this is a hack to get return-type annotation into the 'property'
      dummy_fget.__annotations__ = {
        'return': np.ndarray[(..., *shape), np.dtype[dtype]] }

    self.fget = dummy_fget

  #-----------------------------------------------------------------------------
  def __set_name__(self, owner, name):
    self._name = name

  #-----------------------------------------------------------------------------
  def set_tuple_idx(self, idx):
    self._idx = idx

  #-----------------------------------------------------------------------------
  def __get__( self, obj, objtype = None ):
    if obj is None:
      return self

    return obj._data[self._idx]

  #-----------------------------------------------------------------------------
  def __set__( self, obj, value ):
    if obj is None:
      raise AttributeError(f"Re-defining info property not allowed: {self._name}")

    obj._data[self._idx][:] = value

  #-----------------------------------------------------------------------------
  def __delete__(self, obj):
    raise AttributeError(f"Deleting info property not allowed: {self._name}")

  #-----------------------------------------------------------------------------
  def getter(self, fget):
      raise ValueError(f"Changing getter not allowed")

  #-----------------------------------------------------------------------------
  def setter(self, fset):
      raise ValueError(f"Changing setter not allowed")

  #-----------------------------------------------------------------------------
  def deleter(self, fdel):
      raise ValueError(f"Changing deleter not allowed")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class InfoMeta(ABCMeta):
  """Metaclass for Info, defines fields from class definition
  """
  #-----------------------------------------------------------------------------
  def __new__( mcls,
    name,
    bases,
    namespace ):

    fields = dict()

    for b in bases:
      if isinstance(b, InfoMeta):
        fields.update(b.fields)

    for k,v in namespace.items():
      if isinstance(v, InfoField):
        fields[k] = (v._shape, v._dtype)

    keys = list(fields.keys())

    for i, k in enumerate(keys):
      namespace[k].set_tuple_idx(i)

    tuple_cls = namedtuple(name+'Fields', keys)

    namespace['_fields'] = fields
    namespace['_field_keys'] = keys
    namespace['_tuple_cls'] = tuple_cls

    return super().__new__(
      mcls,
      name,
      bases,
      namespace )

  #-----------------------------------------------------------------------------
  @property
  def fields(self) -> dict[str, tuple[tuple[int], np.dtype]]:
    """Field array specification
    """
    return self._fields

  #-----------------------------------------------------------------------------
  @property
  def tuple_cls(self) -> namedtuple:
    """Named type class used to store field arrays
    """
    return self._tuple_cls

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Info(Sequence, metaclass = InfoMeta):
  """Container for state of AMR
  """

  __slots__ = ('_shape', '_data')

  #-----------------------------------------------------------------------------
  def __init__(self, *args, **kwargs):

    self._shape = None
    self._data = self._tuple_cls(*[None]*len(self._fields))

    if len(args) and len(kwargs):
      raise ValueError(f"Arguments must be only size, or all info values")

    if len(args):
      if len(args) == 1:
        if isinstance(args[0], int):
          self.resize(args[0])
        else:
          self.set_from(args[0])
      else:
        self.set_from(args)
    elif len(kwargs):
      self.set_from(kwargs)

    else:
      self.resize(0)

  #-----------------------------------------------------------------------------
  @property
  def shape(self) -> tuple[int]:
    """Base (common) shape of all field arrays
    """
    return self._shape

  #-----------------------------------------------------------------------------
  @property
  def data(self) -> Info.tuple_cls:
    return self._data

  #-----------------------------------------------------------------------------
  def __len__(self):
    return self._shape[0]

  #-----------------------------------------------------------------------------
  def __getitem__( self, idx ):
    return type(self)( *[arr[idx] for arr in self._data] )

  #-----------------------------------------------------------------------------
  def __setitem__( self, idx, info ):
    if not isinstance(info, (type(self), self._tuple_cls)):
      raise ValueError(f"Expected {type(self).__name__}: {type(info)}")

    if isinstance(info, type(self)):
      info = info._data

    for a, b in zip(self._data, info):
      a[idx] = b

  #-----------------------------------------------------------------------------
  def set_from(self, info : Union[tuple, dict]):
    self._shape, self._data = self._validate_tuple(info)

  #-----------------------------------------------------------------------------
  def _validate_tuple(self, info):
    cls = self._tuple_cls
    fields = self._fields

    if not isinstance(info, cls):
      if isinstance(info, Mapping):
        info = cls(**info)
      else:
        info = cls(*info)

    base_shape = None

    for (k, (shape, dtype)), v in zip(fields.items(), info):
      # number of trailing dimensions that must match the field spec
      ndim = v.ndim - len(shape)

      if v.shape[ndim:] != shape or v.dtype != dtype:
        raise ValueError(f"'{k}' must have trailing shape[{ndim}:] == {shape}, dtype == {dtype}: {v.shape[ndim:]}, {v.dtype}")

      if base_shape is None:
        base_shape = v.shape[:ndim]

      elif v.shape[:ndim] != base_shape:
        raise ValueError(f"All arrays must have same leading shape {base_shape}: {k} -> {v.shape[:ndim]}")

    return base_shape, info

  #-----------------------------------------------------------------------------
  def resize(self, size : int):
    """Resize the first dimension of all field arrays

    Parameters
    ----------
    size :
      New size of first axis
    """
    if self._shape is not None and size == self._shape[0]:
      return

    info = list()

    for (k, (shape, dtype)), _arr in zip(self._fields.items(), self._data):

      arr = np.zeros((size, *shape), dtype = dtype)

      if _arr is not None:
        arr[:len(_arr)] = _arr[:size]

      info.append(arr)

    self.set_from(info)

  #-----------------------------------------------------------------------------
  def contiguous(self) -> Info:
    """Return a new Info object with all contiguous arrays, making a copy
    only when necessary.
    """
    return type(self)( np.ascontiguousarray(arr) for arr in self._data )

  #-----------------------------------------------------------------------------
  def __class_getitem__(cls, *args):
    from types import GenericAlias
    return GenericAlias(cls, args)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class InfoUpdate:
  """Data associated with changes during adaptive mesh refinement
  """

  #-----------------------------------------------------------------------------
  def __init__(self,
    dst,
    src):

    if len(dst) != len(src):
      raise ValueError(f"Adapted data must all be the same: {len(dst)} != {len(src)}")

    self._dst = dst
    self._src = src

  #-----------------------------------------------------------------------------
  def __len__(self):
    return len(self._src)

  #-----------------------------------------------------------------------------
  @property
  def dst(self) -> Info:
    """New leaf info
    """
    return self._dst

  #-----------------------------------------------------------------------------
  @property
  def src(self) -> Info:
    """Replaced (old) leaf info
    """
    return self._src

  #-----------------------------------------------------------------------------
  def __class_getitem__(cls, *args):
    from types import GenericAlias
    return GenericAlias(cls, args)

