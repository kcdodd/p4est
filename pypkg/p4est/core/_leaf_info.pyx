from collections import namedtuple
from collections.abc import (
  Iterable,
  Sequence,
  Mapping )

import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
QUAD_FIELDS = {
  # the index of the original root level mesh.cells
  'root' : (tuple(), np.int32),
  # refinement level
  'level' : (tuple(), np.int8),
  # Normalized coordinate of the leaf's origin relative to the root cell
  # stored as integer units to allow exact arithmetic:
  # 0 -> 0.0
  # 2**max_level -> 1.0
  # To get positions from this origin, the relative width of the leaf can be
  # computed from the refinement level:
  # 2**(max_level - level) -> 1.0/2**level
  # NOTE: This results in higher precision than a normalized 32bit float,
  # since a single-precision float only has 24bits for the fraction.
  # Floating point arithmetic involving the normalized coordinates should
  # use 64bit (double) precision to avoid loss of precision.
  'origin' : ((2,), np.int32),
  # Computational weight of the leaf used for load partitioning among processors
  'weight' : (tuple(), np.int32),
  # A flag used to control refinement (>0) and coarsening (<0)
  'adapt' : (tuple(), np.int8),
  # Indices of up to 6 unique adjacent cells, ordered as:
  #    |110 | 111|
  # ---+----+----+---
  # 001|         |011
  # ---+         +---
  # 000|         |010
  # ---+----+----+---
  #    |100 | 101|
  #
  # Indexing: [(y-normal, x-normal), (-face, +face), (-half, +half)]
  # If cell_adj[i,j,0] == cell_adj[i,j,1], then both halfs are shared with
  # the same cell (of equal or greater size).
  # Otherwise each half is shared with different cells of 1/2 size.
  'cell_adj' : ((2,2,2), np.int32),
  # connecting face {0..7} of the adjacent cell
  #    |   11   |
  # ---+--------+---
  #    |        |
  #  00|        |01
  #    |        |
  # ---+--------+---
  #    |   10   |
  # NOTE: These do not have to be specified for each half like cell_adj, since
  # the adjacent face is the same for both halves since both neighbors must be
  # part of the same tree
  'cell_adj_face' : ((2,2), np.int8),
  # connect sub-face, {0,1} in the case where the adjacent cell is twice the size
  'cell_adj_subface' : ((2,2), np.int8),
  # relative ordering {-1, 1} of the adjacent cell
  'cell_adj_order' : ((2,2), np.int8),
  # relative level {-1, 0, 1} of the adjacent cell
  'cell_adj_level' : ((2,2), np.int8) }

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
QuadInfoTuple = namedtuple('QuadInfoTuple', list(QUAD_FIELDS.keys()))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadInfo:
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
  def as_tuple(self):
    return QuadInfoTuple(
      root = self._root,
      level = self._level,
      origin = self._origin,
      weight = self._weight,
      adapt = self._adapt,
      cell_adj = self._cell_adj,
      cell_adj_face = self._cell_adj_face,
      cell_adj_subface = self._cell_adj_subface,
      cell_adj_order = self._cell_adj_order,
      cell_adj_level = self._cell_adj_level )

  #-----------------------------------------------------------------------------
  def set_from(self, info):
    if not isinstance(info, QuadInfoTuple):
      if isinstance(info, Mapping):
        info = QuadInfoTuple(**info)
      else:
        info = QuadInfoTuple(*info)

    base_shape = None

    for (k, (shape, dtype)), v in zip(QUAD_FIELDS.items(), info):
      ndim = v.ndim - len(shape)

      if v.shape[ndim:] != shape or v.dtype != dtype:
        raise ValueError(f"'{k}' must have trailing shape[{ndim}:] == {shape}, dtype == {dtype}: {v.shape[ndim:]}, {v.dtype}")

      if base_shape is None:
        base_shape = v.shape[:ndim]

      elif v.shape[:ndim] != base_shape:
        raise ValueError(f"All arrays must have same leading shape {base_shape}: {k} -> {v.shape[:ndim]}")

    self._root = np.ascontiguousarray(info.root)
    self._level = np.ascontiguousarray(info.level)
    self._origin = np.ascontiguousarray(info.origin)
    self._weight = np.ascontiguousarray(info.weight)
    self._adapt = np.ascontiguousarray(info.adapt)
    self._cell_adj = np.ascontiguousarray(info.cell_adj)
    self._cell_adj_face = np.ascontiguousarray(info.cell_adj_face)
    self._cell_adj_subface = np.ascontiguousarray(info.cell_adj_subface)
    self._cell_adj_order = np.ascontiguousarray(info.cell_adj_order)
    self._cell_adj_level = np.ascontiguousarray(info.cell_adj_level)

  #-----------------------------------------------------------------------------
  @property
  def root(self):
    return self._root

  #-----------------------------------------------------------------------------
  @root.setter
  def root(self, val):
    self._root[:] = val

  #-----------------------------------------------------------------------------
  @property
  def level(self):
    return self._level

  #-----------------------------------------------------------------------------
  @level.setter
  def level(self, val):
    self._level[:] = val

  #-----------------------------------------------------------------------------
  @property
  def origin(self):
    return self._origin

  #-----------------------------------------------------------------------------
  @origin.setter
  def origin(self, val):
    self._origin[:] = val

  #-----------------------------------------------------------------------------
  @property
  def weight(self):
    return self._weight

  #-----------------------------------------------------------------------------
  @weight.setter
  def weight(self, val):
    self._weight[:] = val

  #-----------------------------------------------------------------------------
  @property
  def adapt(self):
    return self._adapt

  #-----------------------------------------------------------------------------
  @adapt.setter
  def adapt(self, val):
    self._adapt[:] = val

  #-----------------------------------------------------------------------------
  @property
  def cell_adj(self):
    return self. _cell_adj

  #-----------------------------------------------------------------------------
  @cell_adj.setter
  def cell_adj(self, val):
    self._cell_adj[:] = val

  #-----------------------------------------------------------------------------
  @property
  def cell_adj_face(self):
    return self._cell_adj_face

  #-----------------------------------------------------------------------------
  @cell_adj_face.setter
  def cell_adj_face(self, val):
    self._cell_adj_face[:] = val

  #-----------------------------------------------------------------------------
  @property
  def cell_adj_subface(self):
    return self._cell_adj_subface

  #-----------------------------------------------------------------------------
  @cell_adj_subface.setter
  def cell_adj_subface(self, val):
    self._cell_adj_subface[:] = val

  #-----------------------------------------------------------------------------
  @property
  def cell_adj_order(self):
    return self._cell_adj_order

  #-----------------------------------------------------------------------------
  @cell_adj_order.setter
  def cell_adj_order(self, val):
    self._cell_adj_order[:] = val

  #-----------------------------------------------------------------------------
  @property
  def cell_adj_level(self):
    return self._cell_adj_level

  #-----------------------------------------------------------------------------
  @cell_adj_level.setter
  def cell_adj_level(self, val):
    self._cell_adj_level[:] = val

  #-----------------------------------------------------------------------------
  def resize(self, size):
    if self._root is not None and size == len(self._root):
      return

    info = list()

    for (k, (shape, dtype)), _arr in zip(QUAD_FIELDS.items(), self.as_tuple()):

      arr = np.zeros((size, *shape), dtype = dtype)

      if _arr is not None:
        arr[:len(_arr)] = _arr[:size]

      info.append(arr)

    self.set_from(info)

  #-----------------------------------------------------------------------------
  def __len__(self):
    return len(self._root)

  #-----------------------------------------------------------------------------
  def __getitem__( self, idx ):
    return QuadInfo(
      np.ascontiguousarray(arr[idx])
      for arr in self.as_tuple() )

  #-----------------------------------------------------------------------------
  def __setitem__( self, idx, info ):
    if not isinstance(info, (QuadInfo, QuadInfoTuple)):
      raise ValueError(f"Expected QuadInfo: {type(info)}")

    if isinstance(info, QuadInfo):
      info = info.as_tuple()

    for a, b in zip(self.as_tuple(), info):
      a[idx] = b
