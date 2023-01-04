import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class Adapted:
  """Data associated with changes during adaptive mesh refinement
  """

  #-----------------------------------------------------------------------------
  def __init__(self,
    idx,
    info,
    replaced_idx,
    replaced_info):

    if not (
      len(idx) == len(info)
      and len(idx) == len(info)
      and len(idx) == len(replaced_idx)
      and len(idx) == len(replaced_info) ):

      raise ValueError("Adapted data must all be the same length")

    self._idx = np.ascontiguousarray(idx)
    self._info = info.contiguous()
    self._replaced_idx = np.ascontiguousarray(replaced_idx)
    self._replaced_info = replaced_info.contiguous()

  #-----------------------------------------------------------------------------
  def __len__(self):
    return len(self._idx)

  #-----------------------------------------------------------------------------
  @property
  def idx(self):
    """New leaf indices
    """
    return self._idx

  #-----------------------------------------------------------------------------
  @property
  def info(self):
    """New leaf info
    """
    return self._info

  #-----------------------------------------------------------------------------
  @property
  def replaced_idx(self):
    """Replaced (old) leaf indices
    """
    return self._replaced_idx

  #-----------------------------------------------------------------------------
  @property
  def replaced_info(self):
    """Replaced (old) leaf info
    """
    return self._replaced_info

