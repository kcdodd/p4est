cimport numpy as np
from p4est.core._leaf_info cimport QuadInfo

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class Adapted:
  cdef np.ndarray _idx
  cdef QuadInfo _info

  cdef np.ndarray _replaced_idx
  cdef QuadInfo _replaced_info
