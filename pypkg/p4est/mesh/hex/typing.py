from typing import (
  Union,
  Literal,
  TypeVar )
import numpy as np
from ...typing import N, M, NV, NN, NE, NC, NewType
from ...utils import jagged_array

CoordRel = NewType('CoordRel', np.ndarray[(Union[N,Literal[1]], ..., 3), np.dtype[np.floating]])
r"""Relative local coordinates :math:`\in [0.0, 1.0]^3`

Indexing is ``[..., (axis0, axis1, axis2)]``

.. code-block:: python

  local_coord[..., axis0] ⟶ local_coord0
  local_coord[..., axis1] ⟶ local_coord1
  local_coord[..., axis2] ⟶ local_coord2

"""

CoordAbs = NewType('CoordAbs', np.ndarray[(N, ..., 3), np.dtype[np.floating]])
r"""Absolute global coordinates :math:`\in \mathbb{R}^3`

Indexing is ``[..., (axis0, axis1, axis2)]``

.. code-block:: python

  global_coord[..., axis0] ⟶ global_coord0
  global_coord[..., axis1] ⟶ global_coord1
  global_coord[..., axis2] ⟶ global_coord2

.. math::

  \func{\rankone{x}}{\rankone{q}} =
  \begin{bmatrix}
    \func{\rankzero{x_0}}{\rankzero{q}_0, \rankzero{q}_1, \rankzero{q}_2} \\
    \func{\rankzero{x_1}}{\rankzero{q}_0, \rankzero{q}_1, \rankzero{q}_2} \\
    \func{\rankzero{x_2}}{\rankzero{q}_0, \rankzero{q}_1, \rankzero{q}_2}
  \end{bmatrix}

"""

CoordAbsJac = NewType('CoordAbsJac', np.ndarray[(N, ..., 3,3), np.dtype[np.floating]])
r"""Jacobian of the absolute coordinates w.r.t local coordinates

.. math::

  \ranktwo{J}_\rankone{x} = \nabla_{\rankone{q}} \rankone{x} =
  \begin{bmatrix}
    \frac{\partial x_0}{\partial q_0} & \frac{\partial x_0}{\partial q_1} & \frac{\partial x_0}{\partial q_2} \\
    \frac{\partial x_1}{\partial q_0} & \frac{\partial x_1}{\partial q_1} & \frac{\partial x_1}{\partial q_2} \\
    \frac{\partial x_2}{\partial q_0} & \frac{\partial x_2}{\partial q_1} & \frac{\partial x_2}{\partial q_2}
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

Indexing is ``[cell, ∓axis0, ∓axis1, ∓axis2]``

.. code-block::

  cells[:,0,0,0] ⟶ vert(-axis0, -axis1, -axis2)
  cells[:,0,0,1] ⟶ vert(-axis0, -axis1, +axis2)

  cells[:,0,1,0] ⟶ vert(-axis0, +axis1, -axis2)
  cells[:,0,1,1] ⟶ vert(-axis0, +axis1, +axis2)

  cells[:,1,0,0] ⟶ vert(+axis0, -axis1, -axis2)
  cells[:,1,0,1] ⟶ vert(+axis0, -axis1, +axis2)

  cells[:,1,1,0] ⟶ vert(+axis0, +axis1, -axis2)
  cells[:,1,1,1] ⟶ vert(+axis0, +axis1, +axis2)
"""

CellAdj = NewType('CellAdj', np.ndarray[(NC, 3, 2), np.dtype[np.integer]])
r"""Topological connectivity to other cells accross each face,
cell :math:`\in` :class:`~p4est.typing.NC` ⟶ *cell*
:math:`\in` :class:`~p4est.typing.NC`.

Indexing is ``[cell, (axis0, axis1, axis2), ∓{axis0|axis1|axis2}]``

.. code-block::

  cell_adj[:,0,0] ⟶ face_axis0(-axis0)
  cell_adj[:,0,1] ⟶ face_axis0(+axis0)

  cell_adj[:,1,0] ⟶ face_axis1(-axis1)
  cell_adj[:,1,1] ⟶ face_axis1(+axis1)

  cell_adj[:,2,0] ⟶ face_axis2(-axis2)
  cell_adj[:,2,1] ⟶ face_axis2(+axis2)
"""

CellAdjFace = NewType('CellAdjFace', np.ndarray[(NC, 3, 2), np.dtype[np.integer]])
r"""Topological order of the faces of each connected cell,
cell :math:`\in` :class:`~p4est.typing.NC` ⟶ *cell-face* :math:`\in [0,11]`

Indexing is ``[cell, (axis0, axis1, axis2), ∓(axis0|axis1|axis2)]``
"""

CellNodes = NewType('CellNodes', np.ndarray[(NC,2,2,2), np.dtype[np.integer]])
r"""Mapping cell :math:`\in` :class:`~p4est.typing.NC` ⟶ node
:math:`\in` :class:`~p4est.typing.NN`

Indexing is ``[cell, ∓axis0, ∓axis1, ∓axis2]``

.. code-block::

  cell_nodes[:,0,0,0] ⟶ node(-axis0, -axis1, -axis2)
  cell_nodes[:,0,0,1] ⟶ node(-axis0, -axis1, +axis2)

  cell_nodes[:,0,1,0] ⟶ node(-axis0, +axis1, -axis2)
  cell_nodes[:,0,1,1] ⟶ node(-axis0, +axis1, +axis2)

  cell_nodes[:,1,0,0] ⟶ node(+axis0, -axis1, -axis2)
  cell_nodes[:,1,0,1] ⟶ node(+axis0, -axis1, +axis2)

  cell_nodes[:,1,1,0] ⟶ node(+axis0, +axis1, -axis2)
  cell_nodes[:,1,1,1] ⟶ node(+axis0, +axis1, +axis2)

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

Indexing is ``[cell, (axis0, axis1, axis2), ∓{axis1|axis2|axis0}, ∓{axis2|axis0|axis1}]``

.. code-block::

  cell_edges[:,0,0,0] ⟶ edge_axis0(-axis1, -axis2)
  cell_edges[:,0,0,1] ⟶ edge_axis0(-axis1, +axis2)
  cell_edges[:,0,1,0] ⟶ edge_axis0(+axis1, -axis2)
  cell_edges[:,0,1,1] ⟶ edge_axis0(+axis1, +axis2)

  cell_edges[:,1,0,0] ⟶ edge_axis1(-axis2, -axis0)
  cell_edges[:,1,0,1] ⟶ edge_axis1(-axis2, +axis0)
  cell_edges[:,1,1,0] ⟶ edge_axis1(+axis2, -axis0)
  cell_edges[:,1,1,1] ⟶ edge_axis1(+axis2, +axis0)

  cell_edges[:,2,0,0] ⟶ edge_axis2(-axis1, -axis0)
  cell_edges[:,2,0,1] ⟶ edge_axis2(-axis1, +axis0)
  cell_edges[:,2,1,0] ⟶ edge_axis2(+axis1, -axis0)
  cell_edges[:,2,1,1] ⟶ edge_axis2(+axis1, +axis0)
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
