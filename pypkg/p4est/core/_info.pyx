from collections import namedtuple
from collections.abc import (
  Iterable,
  Sequence,
  Mapping )
from functools import lru_cache
import numpy as np


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class Info:
  """Container for state of AMR
  """
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
  def shape(self):
    return self._shape

  #-----------------------------------------------------------------------------
  @property
  def fields(self):
    return self._fields()

  #-----------------------------------------------------------------------------
  @property
  def tuple_cls(self):
    return self._tuple_cls()

  #-----------------------------------------------------------------------------
  @lru_cache(maxsize = 1)
  def _tuple_cls(self):
    return namedtuple(f'{type(self).__name__}Tuple', list(self.fields.keys()))

  #-----------------------------------------------------------------------------
  def __len__(self):
    return self._shape[0]

  #-----------------------------------------------------------------------------
  def __getitem__( self, idx ):
    return type(self)( arr[idx] for arr in self.as_tuple() )

  #-----------------------------------------------------------------------------
  def __setitem__( self, idx, info ):
    if not isinstance(info, (type(self), self._tuple_cls())):
      raise ValueError(f"Expected {type(self).__name__}: {type(info)}")

    if isinstance(info, type(self)):
      info = info.as_tuple()

    for a, b in zip(self.as_tuple(), info):
      a[idx] = b

  #-----------------------------------------------------------------------------
  def as_tuple(self):
    return self.tuple_cls(*[getattr(self,'_'+k) for k in self._fields().keys()])

  #-----------------------------------------------------------------------------
  def set_from(self, info):
    base_shape, info = self._validate_tuple(info)

    for k,v in zip(self._fields().keys(), info):
      setattr(self, '_'+k, v)

    self._shape = base_shape

  #-----------------------------------------------------------------------------
  def _validate_tuple(self, info):
    cls = self.tuple_cls
    fields = self.fields

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

    for (k, (shape, dtype)), _arr in zip(self._fields().items(), self.as_tuple()):

      arr = np.zeros((size, *shape), dtype = dtype)

      if _arr is not None:
        arr[:len(_arr)] = _arr[:size]

      info.append(arr)

    self.set_from(info)

  #-----------------------------------------------------------------------------
  def contiguous(self):
    return type(self)( np.ascontiguousarray(arr) for arr in self.as_tuple() )

  #-----------------------------------------------------------------------------
  def _fields(self):
    raise NotImplementedError

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class CellInfo(Info):
  """AMR cell state
  """
  pass

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadLocalInfo(CellInfo):
  """AMR cells owned by local process
  """
  #-----------------------------------------------------------------------------
  @lru_cache(maxsize = 1)
  def _fields(self):
    return {
      # the index of the original root level mesh.cells
      'root' : (tuple(), np.int32),
      # unique local index
      'idx' : (tuple(), np.int32),
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
      # Indexing: [(x-normal, y-normal), (-face, +face), (-half, +half)]
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
      'cell_adj_level' : ((2,2), np.int8),
      # MPI process rank of adjacent cell
      'cell_adj_rank' : ((2,2,2), np.int32),
      #The nodes that are inside of a cell
      'cell_nodes' : ((2,2), np.int32)
      }

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
  def idx(self):
    return self._idx

  #-----------------------------------------------------------------------------
  @idx.setter
  def idx(self, val):
    self._idx[:] = val

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
    return self._cell_adj

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
  @property
  def cell_adj_rank(self):
    return self._cell_adj_rank

  #-----------------------------------------------------------------------------
  @cell_adj_rank.setter
  def cell_adj_rank(self, val):
    self._cell_adj_rank[:] = val

  #-----------------------------------------------------------------------------
  @property
  def cell_nodes(self):
    return self._cell_nodes

  #-----------------------------------------------------------------------------
  @cell_nodes.setter
  def cell_nodes(self, val):
    self._cell_nodes[:] = val

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadGhostInfo(CellInfo):
  """AMR cells neighboring one or more local cells, but owned by non-local process
  """
  #-----------------------------------------------------------------------------
  def _fields(self):
    return {
      'rank' : (tuple(), np.int32),
      'root' : (tuple(), np.int32),
      'idx' : (tuple(), np.int32),
      'level' : (tuple(), np.int8),
      'origin' : ((2,), np.int32) }

  #-----------------------------------------------------------------------------
  @property
  def rank(self) -> int:
    """Owning rank of cell
    """
    return self._rank

  #-----------------------------------------------------------------------------
  @rank.setter
  def rank(self, val):
    self._rank[:] = val

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
  def idx(self):
    return self._idx

  #-----------------------------------------------------------------------------
  @idx.setter
  def idx(self, val):
    self._idx[:] = val

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
  def origin(self) -> np.ndarray[(...,2,2), np.dtype[np.integer]]:
    """Relative origin of refined cell within root cell
    """
    return self._origin

  #-----------------------------------------------------------------------------
  @origin.setter
  def origin(self, val):
    self._origin[:] = val

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class HexLocalInfo(QuadLocalInfo):
  #-----------------------------------------------------------------------------
  @lru_cache(maxsize = 1)
  def _fields(self):
    return {
      # the index of the original root level mesh.cells
      'root' : (tuple(), np.int32),
      # unique local index
      'idx' : (tuple(), np.int32),
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
      'origin' : ((3,), np.int32),
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
      # Indexing: [(x-normal, y-normal, z-normal), (-face, +face), (-half, +half)]
      # If cell_adj[i,j,0] == cell_adj[i,j,1], then both halfs are shared with
      # the same cell (of equal or greater size).
      # Otherwise each half is shared with different cells of 1/2 size.
      'cell_adj' : ((3,2, 2,2), np.int32),
      # connecting face {0..23} of the adjacent cell
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
      'cell_adj_face' : ((3,2), np.int8),
      # connect sub-face, {0,1} in the case where the adjacent cell is twice the size
      'cell_adj_subface' : ((3,2), np.int8),
      # relative ordering {-1, 1} of the adjacent cell
      'cell_adj_order' : ((3,2), np.int8),
      # relative level {-1, 0, 1} of the adjacent cell
      'cell_adj_level' : ((3,2), np.int8),
      # MPI process rank of adjacent cell
      'cell_adj_rank' : ((3,2, 2,2), np.int32)  }

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class HexGhostInfo(QuadGhostInfo):
  #-----------------------------------------------------------------------------
  def _fields(self):
    return {
      'rank' : (tuple(), np.int32),
      'root' : (tuple(), np.int32),
      'idx' : (tuple(), np.int32),
      'level' : (tuple(), np.int8),
      'origin' : ((3,), np.int32) }

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class NodeInfo(Info):
  #-----------------------------------------------------------------------------
  @lru_cache(maxsize = 1)
  def _fields(self):
    return {
      # unique local index
      'idx' : (tuple(), np.int32),
      # Cells that are connected to this node
      'cells' : ((2,2), np.int8),
      # Inverse order of cells connected to this node
      'cells_inv' : ((2,2), np.int8)  }

  #-----------------------------------------------------------------------------
  @property
  def idx(self):
    return self._idx

  #-----------------------------------------------------------------------------
  @idx.setter
  def idx(self, val):
    self._idx[:] = val

  #-----------------------------------------------------------------------------
  @property
  def cells(self):
    return self._cells

  #-----------------------------------------------------------------------------
  @cells.setter
  def cells(self, val):
    self._cells[:] = val

  #-----------------------------------------------------------------------------
  @property
  def cells_inv(self):
    return self._cells_inv

  #-----------------------------------------------------------------------------
  @cells_inv.setter
  def cells_inv(self, val):
    self._cells_inv[:] = val