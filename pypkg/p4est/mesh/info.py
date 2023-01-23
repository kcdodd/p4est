# Enable postponed evaluation of annotations
from __future__ import annotations
from partis.utils import TYPING

if TYPING:
  from typing import (
    Union,
    Literal,
    get_type_hints)
  from ..typing import (
    NP,
    NTX,
    NRX,
    NL,
    NG,
    NM,
    NAM,
    NAF,
    NAC,
    Where )

from collections import namedtuple
from collections.abc import (
  Iterable,
  Sequence,
  Mapping )
from functools import lru_cache
import numpy as np

from abc import ABCMeta
from typing import NamedTuple

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class InfoField:
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
  def fields(self):
    return self._fields

  #-----------------------------------------------------------------------------
  @property
  def tuple_cls(self):
    return self._tuple_cls

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Info(Sequence, metaclass = InfoMeta):
  """Container for state of AMR
  """

  __slots__ = ('_shape', '_data')

  #-----------------------------------------------------------------------------
  def __init__(self, *args, **kwargs):

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
  def data(self):
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
  def set_from(self, info):
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
  def resize(self, size):
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
  def contiguous(self):
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



