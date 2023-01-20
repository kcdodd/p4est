import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class QuadAdapted:
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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class PartitionedLocal:
  """
  """
  #-----------------------------------------------------------------------------
  def __init__(self,
    from_idx,
    to_idx):

    if not (
      len(from_idx) == len(to_idx)):

      raise ValueError("Partition data must all be the same length")

    self._from_idx = np.ascontiguousarray(from_idx)
    self._to_idx = np.ascontiguousarray(to_idx)

  #-----------------------------------------------------------------------------
  def __len__(self):
    return len(self._from_idx)

  #-----------------------------------------------------------------------------
  @property
  def from_idx(self):
    """Previous leaf indices
    """
    return self._from_idx

  #-----------------------------------------------------------------------------
  @property
  def to_idx(self):
    """New leaf indices
    """
    return self._to_idx

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class PartitionedRemote:
  """
  """

  #-----------------------------------------------------------------------------
  def __init__(self,
    local_idx,
    remote_idx,
    rank):

    if not (
      len(local_idx) == len(remote_idx)
      and len(remote_idx) == len(rank) ):

      raise ValueError("Partition data must all be the same length")

    self._local_idx = np.ascontiguousarray(local_idx)
    self._remote_idx = np.ascontiguousarray(remote_idx)
    self._rank = np.ascontiguousarray(rank)

  #-----------------------------------------------------------------------------
  def __len__(self):
    return len(self._local_idx)

  #-----------------------------------------------------------------------------
  @property
  def local_idx(self):
    """Local leaf indices
    """
    return self._local_idx

  #-----------------------------------------------------------------------------
  @property
  def remote_idx(self):
    """Remote leaf indices
    """
    return self._remote_idx

  #-----------------------------------------------------------------------------
  @property
  def rank(self):
    """Remote owner rank
    """
    return self._rank