from typing import (
  Union,
  Literal,
  TypeVar )
import numpy as np
from ...typing import N, M, NV, NN, NE, NC, NewType
from ...utils import jagged_array

CoordRel = NewType('CoordRel', np.ndarray[(Union[N,Literal[1]], ..., 3), np.dtype[np.floating]])
r"""Relative coordinates :math:`\in [0.0, 1.0]^3`
"""

CoordAbs = NewType('CoordAbs', np.ndarray[(N, ..., 3), np.dtype[np.floating]])
r"""Absolute coordinates :math:`\in \mathbb{R}^3`

.. math::

  \func{\rankone{r}}{\rankone{q}} =
  \begin{bmatrix}
    \func{\rankzero{x}}{\rankzero{q}_0, \rankzero{q}_1, \rankzero{q}_2} \\
    \func{\rankzero{y}}{\rankzero{q}_0, \rankzero{q}_1, \rankzero{q}_2} \\
    \func{\rankzero{z}}{\rankzero{q}_0, \rankzero{q}_1, \rankzero{q}_2}
  \end{bmatrix}

"""

CoordAbsJac = NewType('CoordAbsJac', np.ndarray[(N, ..., 3,3), np.dtype[np.floating]])
r"""Jacobian of the absolute coordinates w.r.t local coordinates

.. math::

  \ranktwo{J}_\rankone{r} = \nabla_{\rankone{q}} \rankone{r} =
  \begin{bmatrix}
    \frac{\partial x}{\partial q_0} & \frac{\partial x}{\partial q_1} & \frac{\partial x}{\partial q_2} \\
    \frac{\partial y}{\partial q_0} & \frac{\partial y}{\partial q_1} & \frac{\partial y}{\partial q_2} \\
    \frac{\partial z}{\partial q_0} & \frac{\partial z}{\partial q_1} & \frac{\partial z}{\partial q_2}
  \end{bmatrix}
"""

CoordGeom = NewType('CoordGeom', np.ndarray[(Union[N,Literal[1]], ..., 2,2,2,3), np.dtype[np.floating]])
r"""Absolute coordinates :math:`\in \mathbb{R}^3` at 8 vertices

See Also
--------
* :class:`Cells` for indexing
"""

Vertices = NewType('Vertices', np.ndarray[(NV, 3), np.dtype[np.floating]])
r"""Mapping vertex :math:`\in` :class:`~p4est.typing.NV` ⟶ position :math:`\in \mathbb{R}^3`
"""

VertNodes = NewType('VertNodes', np.ndarray[(NV,), np.dtype[np.integer]])
r"""Mapping vertex :math:`\in` :class:`~p4est.typing.NV` ⟶ node
:math:`\in` :class:`~p4est.typing.NN`
"""

VertGeom = NewType('VertGeom', np.ndarray[(NV,), np.dtype[np.integer]])
r"""Mapping vertex :math:`\in` :class:`~p4est.typing.NV` ⟶ geometry
"""

Cells = NewType('Cells', np.ndarray[(NC, 2, 2, 2), np.dtype[np.integer]])
r"""Mapping cell :math:`\in` :class:`~p4est.typing.NC` ⟶ vertex
:math:`\in` :class:`~p4est.typing.NV`.

Indexing is ``[cell, ∓z, ∓y, ∓x]``

.. code-block::

  cells[:,0,0,0] ⟶ vert(-z, -y, -x)
  cells[:,0,0,1] ⟶ vert(-z, -y, +x)

  cells[:,0,1,0] ⟶ vert(-z, +y, -x)
  cells[:,0,1,1] ⟶ vert(-z, +y, +x)

  cells[:,1,0,0] ⟶ vert(+z, -y, -x)
  cells[:,1,0,1] ⟶ vert(+z, -y, +x)

  cells[:,1,1,0] ⟶ vert(+z, +y, -x)
  cells[:,1,1,1] ⟶ vert(+z, +y, +x)
"""

CellAdj = NewType('CellAdj', np.ndarray[(NC, 3, 2), np.dtype[np.integer]])
r"""Topological connectivity to other cells accross each face,
cell :math:`\in` :class:`~p4est.typing.NC` ⟶ *cell*
:math:`\in` :class:`~p4est.typing.NC`.

Indexing is ``[cell, (x,y,z), ∓(x|y|z)]``

.. code-block::

  cell_adj[:,0,0] ⟶ xface(-x)
  cell_adj[:,0,1] ⟶ xface(+x)

  cell_adj[:,1,0] ⟶ yface(-y)
  cell_adj[:,1,1] ⟶ yface(+y)

  cell_adj[:,2,0] ⟶ zface(-z)
  cell_adj[:,2,1] ⟶ zface(+z)
"""

CellAdjFace = NewType('CellAdjFace', np.ndarray[(NC, 3, 2), np.dtype[np.integer]])
r"""Topological order of the faces of each connected cell,
cell :math:`\in` :class:`~p4est.typing.NC` ⟶ *cell-face* :math:`\in [0,11]`

Indexing is ``[cell, (x,y,z), ∓(x|y|z)]``
"""

CellNodes = NewType('CellNodes', np.ndarray[(NC,2,2,2), np.dtype[np.integer]])
r"""Mapping cell :math:`\in` :class:`~p4est.typing.NC` ⟶ node
:math:`\in` :class:`~p4est.typing.NN`

Indexing is ``[cell, ∓z, ∓y, ∓x]``

.. code-block::

  cell_nodes[:,0,0,0] ⟶ node(-z, -y, -x)
  cell_nodes[:,0,0,1] ⟶ node(-z, -y, +x)

  cell_nodes[:,0,1,0] ⟶ node(-z, +y, -x)
  cell_nodes[:,0,1,1] ⟶ node(-z, +y, +x)

  cell_nodes[:,1,0,0] ⟶ node(+z, -y, -x)
  cell_nodes[:,1,0,1] ⟶ node(+z, -y, +x)

  cell_nodes[:,1,1,0] ⟶ node(+z, +y, -x)
  cell_nodes[:,1,1,1] ⟶ node(+z, +y, +x)
"""

NodeCells = NewType('NodeCells', jagged_array[NN, np.ndarray[M, np.dtype[np.integer]]])
r"""Mapping node :math:`\in` :class:`~p4est.typing.NN` ⟶ cell
:math:`\in` :class:`~p4est.typing.NC`

Indexing is ``[node, cell]``
"""

NodeCellsInv = NewType('NodeCellsInv', jagged_array[NN, np.ndarray[M, np.dtype[np.integer]]])
r"""Mapping node :math:`\in` :class:`~p4est.typing.NN` ⟶ *cell-node* :math:`\in [0,7]`

Indexing is ``[node, cell]``
"""

NodeEdgeCounts = NewType('NodeEdgeCounts', np.ndarray[NN, np.dtype[np.integer]])
r"""Number of edges incident upon each node ("valence" of node).
"""

CellEdges = NewType('CellEdges', np.ndarray[(NC, 3, 2, 2), np.dtype[np.integer]])
r"""Mapping cell :math:`\in` :class:`~p4est.typing.NC` ⟶ edge :math:`\in` :class:`~p4est.typing.NE`

Indexing is ``[cell, (x,y,z), ∓(z|z|y), ∓(y|x|x)]``

.. code-block::

  cell_edges[:,0,0,0] ⟶ xedge(-z, -y)
  cell_edges[:,0,0,1] ⟶ xedge(-z, +y)
  cell_edges[:,0,1,0] ⟶ xedge(+z, -y)
  cell_edges[:,0,1,1] ⟶ xedge(+z, +y)

  cell_edges[:,1,0,0] ⟶ yedge(-z, -x)
  cell_edges[:,1,0,1] ⟶ yedge(-z, +x)
  cell_edges[:,1,1,0] ⟶ yedge(+z, -x)
  cell_edges[:,1,1,1] ⟶ yedge(+z, +x)

  cell_edges[:,2,0,0] ⟶ zedge(-y, -x)
  cell_edges[:,2,0,1] ⟶ zedge(-y, +x)
  cell_edges[:,2,1,0] ⟶ zedge(+y, -x)
  cell_edges[:,2,1,1] ⟶ zedge(+y, +x)
"""

EdgeCells = NewType('EdgeCells', jagged_array[NE, np.ndarray[M, np.dtype[np.integer]]])
r"""Mapping edge :math:`\in` :class:`~p4est.typing.NE` ⟶ cell :math:`\in` :class:`~p4est.typing.NC`

Indexing is ``[edge, cell]``
"""

EdgeCellsInv = NewType('EdgeCellsInv', jagged_array[NE, np.ndarray[M, np.dtype[np.integer]]])
r"""Mapping edge :math:`\in` :class:`~p4est.typing.NE` ⟶ *cell-edge* :math:`\in [0,12]`

Indexing is ``[edge, cell]``
"""

EdgeCellCounts = NewType('EdgeCellCounts', np.ndarray[NE, np.dtype[np.integer]])
r"""Number of cells incident upon each edge ("valence" of edge).
"""
