from typing import (
  Optional,
  Union,
  Literal,
  TypeVar,
  NewType )
import numpy as np
from ...typing import N, M, NV, NN, NE, NC, NewType
from ...utils import jagged_array


Where = NewType('Where', Union[slice, np.ndarray[..., np.dtype[Union[np.integer, bool]]]])
"""Array indexing or boolean mask
"""

CoordRel = NewType('CoordRel', np.ndarray[(Union[N,Literal[1]], ..., 3), np.dtype[np.floating]])
r"""Relative coordinates :math:`\in [0.0, 1.0]^3`
"""

CoordAbs = NewType('CoordAbs', np.ndarray[(N, ..., 3), np.dtype[np.floating]])
r"""Absolute coordinates :math:`\in \mathbb{R}^3`
"""

Vertices = NewType('Vertices', np.ndarray[(NV, 3), np.dtype[np.floating]])
"""Hex mesh vertex positions
"""

Cells = NewType('Cells', np.ndarray[(NC, 2, 2, 2), np.dtype[np.integer]])
"""Hex mesh mapping cells -> vertices

Indexing is ``[cell, ∓z, ∓y, ∓x]``

.. code-block::

  cells[:,0,0,0] -> vert(-z, -y, -x)
  cells[:,0,0,1] -> vert(-z, -y, +x)

  cells[:,0,1,0] -> vert(-z, +y, -x)
  cells[:,0,1,1] -> vert(-z, +y, +x)

  cells[:,1,0,0] -> vert(+z, -y, -x)
  cells[:,1,0,1] -> vert(+z, -y, +x)

  cells[:,1,1,0] -> vert(+z, +y, -x)
  cells[:,1,1,1] -> vert(+z, +y, +x)
"""

CellAdj = NewType('CellAdj', np.ndarray[(NC, 3, 2), np.dtype[np.integer]])
"""Hex mesh mapping cells -> cell

Indexing is ``[cell, (x,y,z), ∓(x|y|z)]``

.. code-block::

  cell_adj[:,0,0] -> xface(-x)
  cell_adj[:,0,1] -> xface(+x)

  cell_adj[:,1,0] -> yface(-y)
  cell_adj[:,1,1] -> yface(+y)

  cell_adj[:,2,0] -> zface(-z)
  cell_adj[:,2,1] -> zface(+z)
"""

CellAdjFace = NewType('CellAdjFace', np.ndarray[(NC, 3, 2), np.dtype[np.integer]])
"""Hex mesh mapping cells -> cell -> face

Indexing is ``[cell, (x,y,z), ∓(x|y|z)]``
"""

VertNodes = NewType('VertNodes', np.ndarray[(NV,), np.dtype[np.integer]])
"""Hex mesh mapping vertices -> nodes
"""

VertGeom = NewType('VertGeom', np.ndarray[(NV,), np.dtype[np.integer]])
"""Hex mesh mapping vertices -> geometry
"""

CellNodes = NewType('CellNodes', np.ndarray[(NC,2,2,2), np.dtype[np.integer]])
"""Hex mesh mapping cells -> nodes

Indexing is ``[cell, ∓z, ∓y, ∓x]``

.. code-block::

  cell_nodes[:,0,0,0] -> node(-z, -y, -x)
  cell_nodes[:,0,0,1] -> node(-z, -y, +x)

  cell_nodes[:,0,1,0] -> node(-z, +y, -x)
  cell_nodes[:,0,1,1] -> node(-z, +y, +x)

  cell_nodes[:,1,0,0] -> node(+z, -y, -x)
  cell_nodes[:,1,0,1] -> node(+z, -y, +x)

  cell_nodes[:,1,1,0] -> node(+z, +y, -x)
  cell_nodes[:,1,1,1] -> node(+z, +y, +x)
"""

NodeCells = NewType('NodeCells', jagged_array[NN, np.ndarray[M, np.dtype[np.integer]]])
"""Hex mesh mapping nodes -> cells

Indexing is ``[node, cell]``
"""

NodeCellsInv = NewType('NodeCellsInv', jagged_array[NN, np.ndarray[M, np.dtype[np.integer]]])
"""Hex mesh mapping nodes -> cells -> node

Indexing is ``[node, cell]``
"""

NodeEdgeCounts = NewType('NodeEdgeCounts', np.ndarray[NN, np.dtype[np.integer]])
"""Number of edges incident upon each node.
"""

CellEdges = NewType('CellEdges', np.ndarray[(NC, 3, 2, 2), np.dtype[np.integer]])
"""Hex mesh mapping cells -> edges

Indexing is ``[cell, (x,y,z), ∓(z|z|y), ∓(y|x|x)]``

.. code-block::

  cell_edges[:,0,0,0] -> xedge(-z, -y)
  cell_edges[:,0,0,1] -> xedge(-z, +y)
  cell_edges[:,0,1,0] -> xedge(+z, -y)
  cell_edges[:,0,1,1] -> xedge(+z, +y)

  cell_edges[:,1,0,0] -> yedge(-z, -x)
  cell_edges[:,1,0,1] -> yedge(-z, +x)
  cell_edges[:,1,1,0] -> yedge(+z, -x)
  cell_edges[:,1,1,1] -> yedge(+z, +x)

  cell_edges[:,2,0,0] -> zedge(-y, -x)
  cell_edges[:,2,0,1] -> zedge(-y, +x)
  cell_edges[:,2,1,0] -> zedge(+y, -x)
  cell_edges[:,2,1,1] -> zedge(+y, +x)
"""

EdgeCells = NewType('EdgeCells', jagged_array[NE, np.ndarray[M, np.dtype[np.integer]]])
"""Hex mesh mapping edges -> cells

Indexing is ``[edge, cell]``
"""

EdgeCellsInv = NewType('EdgeCellsInv', jagged_array[NE, np.ndarray[M, np.dtype[np.integer]]])
"""Hex mesh mapping edges -> cells -> edge

Indexing is ``[edge, cell]``
"""

EdgeCellCounts = NewType('EdgeCellCounts', np.ndarray[NE, np.dtype[np.integer]])
"""Number of cells incident upon each edge.
"""