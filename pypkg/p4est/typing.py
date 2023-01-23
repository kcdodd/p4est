from typing import (
  Union,
  Literal )
import numpy as np
from partis.utils.typing import NewType

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Where = NewType('Where', Union[slice, np.ndarray[..., np.dtype[Union[np.integer, bool]]]])
"""Array indexing or boolean mask
"""

N = NewType('N', int)
"""A variable size
"""

M = NewType('M', int)
"""A variable size
"""

NV = NewType('NV', int)
"""A variable number of vertices
"""

NN = NewType('NN', int)
"""A variable number of nodes
"""

NE = NewType('NE', int)
"""A variable number of edges
"""

NC = NewType('NC', int)
"""A variable number of cells
"""

NP = NewType('NP', int)
"""A variable number of processes in MPI.Comm
"""

NL = NewType('NL', int)
"""A variable number of *local* values on current process rank.
"""

_NL = NewType('_NL', int)
"""A previous number of *local* values on current process rank.
"""

NG = NewType('NG', int)
"""A variable number of *non-local* ghost values belonging to another process that
are "adjacent" to one or more values on the current process rank.
"""

NM = NewType('NM', int)
"""A variable number of *local* mirror values on the boundary with another process.
"""

NAM = NewType('NAM', int)
"""A variable number of adaptive *moved* values within the same rank.
"""

NAC = NewType('NAC', int)
"""A variable number of adaptive *coarse* values within the same rank.
"""

NAF = NewType('NAF', int)
"""A variable number of adaptive *fine* values within the same rank.
"""

NTX = NewType('NTX', int)
"""A variable number of values to be sent to another process rank.
"""

NRX = NewType('NRX', int)
"""A variable number of values to be received from another process rank.
"""