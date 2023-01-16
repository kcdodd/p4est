cimport numpy as np
from p4est.core._leaf_info cimport QuadLocalInfo, HexLocalInfo

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadAdapted:
  cdef np.ndarray _idx
  cdef QuadLocalInfo _info

  cdef np.ndarray _replaced_idx
  cdef QuadLocalInfo _replaced_info

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class HexAdapted:
  cdef np.ndarray _idx
  cdef HexLocalInfo _info

  cdef np.ndarray _replaced_idx
  cdef HexLocalInfo _replaced_info
