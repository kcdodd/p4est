import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class jagged_array:
  """Container for jagged (ragged) array

  Eeach 'row' potentially has a different number of entries, but also each row
  is contiguous (not a general 'sparse' array).
  The array is stored as a single contiguous array of one less dimension.
  Similar to compressed sparse row format, except that 'column' indices are not
  explicitly stored.

  Parameters
  ----------
  data : array of shape = (N, ...)
  row_idx : array of shape = (nrows + 1,)
    Indices of each row, where row ``i`` is ``data[row_idx[i]:row_idx[i+1]]``,
    with ``row_idx[0] == 0`` and ``row_idx[-1] == len(data)``.

  """
  #-----------------------------------------------------------------------------
  def __init__(self, data, row_idx):

    row_idx = np.ascontiguousarray(row_idx)

    if not np.issubdtype(row_idx.dtype, np.integer):
      raise ValueError(f"Must have integral row_idx.dtype: {row_idx.dtype}")

    if (
      row_idx.ndim != 1
      or len(row_idx) == 0 ):

      raise ValueError(f"Must have row_idx.ndim = 1 and len(row_idx) > 0: {row_idx.ndim}, {len(row_idx)}")

    if row_idx[0] != 0:
      raise ValueError(f"Must have row_idx[0] = 0: {row_idx[0]}")

    if row_idx[-1] != len(data):
      raise ValueError(f"Must have row_idx[-1] = len(data) = {len(data)}: {row_idx[-1]}")

    row_counts = np.diff(row_idx)

    if not np.all(row_counts >= 0):
      raise ValueError(f"Must have row_idx[i] <= row_idx[i+i]")


    self._data = data
    self._row_idx = row_idx
    self._row_counts = row_counts

  #-----------------------------------------------------------------------------
  @property
  def flat(self):
    return self._data

  #-----------------------------------------------------------------------------
  @property
  def row_idx(self):
    return self._row_idx

  #-----------------------------------------------------------------------------
  @property
  def row_counts(self):
    return self._row_counts

  #-----------------------------------------------------------------------------
  @property
  def dtype(self):
    return self.data.dtype

  #-----------------------------------------------------------------------------
  def __len__(self):
    return len(self.row_idx) - 1

  #-----------------------------------------------------------------------------
  def __iter__(self):
    for i in range(len(self.row_idx) - 1):
      yield self[i]

  #-----------------------------------------------------------------------------
  def __getitem__( self, idx ):
    return self.data[self.row_idx[idx]:self.row_idx[idx+1]]

  #-----------------------------------------------------------------------------
  def __setitem__( self, idx, row_data ):
    self.data[self.row_idx[idx]:self.row_idx[idx+1]] = row_data
