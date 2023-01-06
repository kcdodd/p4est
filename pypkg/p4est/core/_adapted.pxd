cimport numpy as np
from p4est.core._leaf_info cimport QuadInfo, HexInfo

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadAdapted:
  cdef np.ndarray _idx
  cdef QuadInfo _info

  cdef np.ndarray _replaced_idx
  cdef QuadInfo _replaced_info

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class HexAdapted:
  cdef np.ndarray _idx
  cdef HexInfo _info

  cdef np.ndarray _replaced_idx
  cdef HexInfo _replaced_info
