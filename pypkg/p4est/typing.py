from typing import (
  Optional,
  Union,
  Literal,
  TypeVar )

from partis.utils.typing import NewType

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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