cimport numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class LeafInfo:
  cdef tuple _shape

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadInfo(LeafInfo):
  cdef public np.ndarray _root
  cdef public np.ndarray _level
  cdef public np.ndarray _origin
  cdef public np.ndarray _weight
  cdef public np.ndarray _adapt
  cdef public np.ndarray _cell_adj
  cdef public np.ndarray _cell_adj_face
  cdef public np.ndarray _cell_adj_subface
  cdef public np.ndarray _cell_adj_order
  cdef public np.ndarray _cell_adj_level
  cdef public np.ndarray _cell_adj_rank

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadGhostInfo(LeafInfo):
  cdef public np.ndarray _rank
  cdef public np.ndarray _root
  cdef public np.ndarray _idx
  cdef public np.ndarray _level
  cdef public np.ndarray _origin