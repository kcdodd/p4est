import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class Adapted:
  #-----------------------------------------------------------------------------
  def __init__(self,
    idx,
    info,
    replaced_idx,
    replaced_info):

    self._idx = np.ascontiguousarray(idx)
    self._info = info.contiguous()
    self._replaced_idx = np.ascontiguousarray(replaced_idx)
    self._replaced_info = replaced_info.contiguous()

  #-----------------------------------------------------------------------------
  @property
  def idx(self):
    return self._idx

  #-----------------------------------------------------------------------------
  @property
  def info(self):
    return self._info

  #-----------------------------------------------------------------------------
  @property
  def replaced_idx(self):
    return self._replaced_idx

  #-----------------------------------------------------------------------------
  @property
  def replaced_info(self):
    return self._replaced_info

