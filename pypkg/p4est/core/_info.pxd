cimport numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class Info:
  cdef tuple _shape

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class CellInfo(Info):
  pass

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadLocalInfo(CellInfo):
  cdef public _root
  cdef public _idx
  cdef public _level
  cdef public _origin
  cdef public _weight
  cdef public _adapt
  cdef public _cell_adj
  cdef public _cell_adj_face
  cdef public _cell_adj_subface
  cdef public _cell_adj_order
  cdef public _cell_adj_level
  cdef public _cell_adj_rank
  cdef public _cell_nodes

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadGhostInfo(CellInfo):
  cdef public _rank
  cdef public _root
  cdef public _idx
  cdef public _level
  cdef public _origin


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class HexLocalInfo(QuadLocalInfo):
  pass

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class HexGhostInfo(QuadGhostInfo):
  pass

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class NodelInfo(Info):
  cdef public _idx
  cdef public _cells
  cdef public _cells_inv