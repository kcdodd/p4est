cimport numpy as np
from p4est.core._info cimport QuadLocalInfo, HexLocalInfo

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadAdapted:
  cdef QuadLocalInfo _dst
  cdef QuadLocalInfo _src

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class HexAdapted:
  cdef HexLocalInfo _dst
  cdef HexLocalInfo _src