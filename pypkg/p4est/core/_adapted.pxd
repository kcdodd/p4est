cimport numpy as np
from p4est.core._info cimport QuadLocalInfo, HexLocalInfo

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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class PartitionedLocal:
  cdef np.ndarray _from_idx
  cdef np.ndarray _to_idx

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class PartitionedRemote:
  cdef np.ndarray _local_idx
  cdef np.ndarray _remote_idx
  cdef np.ndarray _rank