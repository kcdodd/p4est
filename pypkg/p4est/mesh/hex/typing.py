from typing import (
  Optional,
  Union,
  Literal,
  TypeVar,
  NewType )
import numpy as np
from ...typing import N, M, NV, NN, NE, NC
from ...utils import jagged_array

Where = Union[slice, np.ndarray[..., np.dtype[Union[np.integer, bool]]]]

#: Relative coordinates
CoordRel = np.ndarray[(Union[N,Literal[1]], ..., 3), np.dtype[np.floating]]

#: Absolute coordinates
CoordAbs = np.ndarray[(N, ..., 3), np.dtype[np.floating]]


#: Hex mesh vertex positions
Vertices = np.ndarray[(NV, 3), np.dtype[np.floating]]

#: Hex mesh mapping cells -> vertices
Cells = np.ndarray[(NC, 2, 2, 2), np.dtype[np.integer]]

#: Hex mesh mapping cells -> cell
CellAdj = np.ndarray[(NC, 3, 2), np.dtype[np.integer]]

#: Hex mesh mapping cells -> cell -> face
CellAdjFace = np.ndarray[(NC, 3, 2), np.dtype[np.integer]]

#: Hex mesh mapping vertices -> nodes
VertNodes = np.ndarray[(NV,), np.dtype[np.integer]]

#: Hex mesh mapping vertices -> geometry
VertGeom = np.ndarray[(NV,), np.dtype[np.integer]]

#: Hex mesh mapping cells -> nodes
CellNodes = np.ndarray[(NC,2,2,2), np.dtype[np.integer]]

#: Hex mesh mapping nodes -> cells
NodeCells = jagged_array[NN, np.ndarray[M, np.dtype[np.integer]]]

#: Hex mesh mapping nodes -> cells -> node
NodeCellsInv = jagged_array[NN, np.ndarray[M, np.dtype[np.integer]]]

# Number of edges incident upon each node.
NodeEdgeCounts = np.ndarray[NN, np.dtype[np.integer]]

#: Hex mesh mapping cells -> edges
CellEdges = np.ndarray[(NC, 3, 2, 2), np.dtype[np.integer]]

#: Hex mesh mapping edges -> cells
EdgeCells = jagged_array[NE, np.ndarray[M, np.dtype[np.integer]]]

#: Hex mesh mapping edges -> cells -> edge
EdgeCellsInv = jagged_array[NE, np.ndarray[M, np.dtype[np.integer]]]

#: Number of cells incident upon each edge.
EdgeCellCounts = np.ndarray[NE, np.dtype[np.integer]]