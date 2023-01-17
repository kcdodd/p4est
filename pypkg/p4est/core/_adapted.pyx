import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadAdapted:
  """__init__(idx, info, replaced_idx, replaced_info)

  Data associated with changes during adaptive mesh refinement

  .. partis_attr:: idx
    :prefix: property
    :type: numpy.ndarray
    :subscript: shape = (num. coarse), dtype = int32

    Coarser level indices into :attr:`P4est.local` (*previous* values, if refining).

  .. partis_attr:: info
    :prefix: property
    :type: QuadLocalInfo
    :subscript: shape = (num. coarse,)

    Coarser level local info, ``local[idx]`` (*previous* values, if refining).

  .. partis_attr:: replaced_idx
    :prefix: property
    :type: numpy.ndarray
    :subscript: shape = (NC,2,2), dtype = int32

    Finer level indices into :attr:`P4est.local` (*previous* values, if coarsening)

  .. partis_attr:: info
    :prefix: property
    :type: QuadLocalInfo
    :subscript: shape = (num. coarse,)

    Coarser level local info, ``local[idx]`` (*previous* values, if refining).
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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class HexAdapted:
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
