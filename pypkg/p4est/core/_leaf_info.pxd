cimport numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadInfo:
  cdef np.ndarray _root
  cdef np.ndarray _level
  cdef np.ndarray _origin
  cdef np.ndarray _weight
  cdef np.ndarray _adapt
  cdef np.ndarray _cell_adj
  cdef np.ndarray _cell_adj_face
  cdef np.ndarray _cell_adj_subface
  cdef np.ndarray _cell_adj_order
  cdef np.ndarray _cell_adj_level
