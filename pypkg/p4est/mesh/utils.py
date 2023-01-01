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
    data = np.ascontiguousarray(data)
    row_idx = np.ascontiguousarray(row_idx)

    if (
      row_idx.ndim != 1
      or len(row_idx) == 0
      or row_idx[0] != 0
      or row_idx[-1] != len(data) ):

      raise ValueError(f"Must have row_idx[0] = 0, and row_idx[-1] = len(data)")

    if not np.all(np.diff(row_idx) > 0):
      raise ValueError(f"row_idx must be monotonically increasing, row_idx[i] < row_idx[i+i]")


    self._data = data
    self._row_idx = row_idx

  #-----------------------------------------------------------------------------
  @property
  def data(self):
    return self._data

  #-----------------------------------------------------------------------------
  @property
  def row_idx(self):
    return self._row_idx

  #-----------------------------------------------------------------------------
  @property
  def shape(self):
    return (len(self.row_idx) - 1, None, *self.data.shape[1:])

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
      yield self.data[self.row_idx[i]:self.row_idx[i+1]]