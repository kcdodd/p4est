import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadAdapted:
  """Data associated with changes during adaptive mesh refinement
  """

  #-----------------------------------------------------------------------------
  def __init__(self,
    dst,
    src):

    if len(dst) != len(src):
      raise ValueError(f"Adapted data must all be the same: {len(dst)} != {len(src)}")

    self._dst = dst.contiguous()
    self._src = src.contiguous()

  #-----------------------------------------------------------------------------
  def __len__(self):
    return len(self._src)

  #-----------------------------------------------------------------------------
  @property
  def dst(self):
    """New leaf info
    """
    return self._dst

  #-----------------------------------------------------------------------------
  @property
  def src(self):
    """Replaced (old) leaf info
    """
    return self._src

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class HexAdapted:
  """Data associated with changes during adaptive mesh refinement
  """
  #-----------------------------------------------------------------------------
  def __init__(self,
    dst,
    src):

    if len(dst) != len(src):
      raise ValueError(f"Adapted data must all be the same: {len(dst)} != {len(src)}")

    self._dst = dst.contiguous()
    self._src = src.contiguous()

  #-----------------------------------------------------------------------------
  def __len__(self):
    return len(self._src)

  #-----------------------------------------------------------------------------
  @property
  def dst(self):
    """New leaf info
    """
    return self._dst

  #-----------------------------------------------------------------------------
  @property
  def src(self):
    """Replaced (old) leaf info
    """
    return self._src
