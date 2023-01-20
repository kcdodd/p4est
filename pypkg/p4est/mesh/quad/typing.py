from typing import (
  Union,
  Literal )
import numpy as np
from ...typing import N, M, NV, NN, NC, NewType
from ...utils import jagged_array

CoordRel = NewType('CoordRel', np.ndarray[(Union[N,Literal[1]], ..., 2), np.dtype[np.floating]])
r"""Relative coordinates :math:`\in [0.0, 1.0]^2`
"""

CoordAbs = NewType('CoordAbs', np.ndarray[(N, ..., 3), np.dtype[np.floating]])
r"""Absolute coordinates :math:`\in \mathbb{R}^3`

.. math::

  \func{\rankone{r}}{\rankone{q}} =
  \begin{bmatrix}
    \func{\rankzero{x}}{\rankzero{q}_0, \rankzero{q}_1} \\
    \func{\rankzero{y}}{\rankzero{q}_0, \rankzero{q}_1} \\
    \func{\rankzero{z}}{\rankzero{q}_0, \rankzero{q}_1}
  \end{bmatrix}


"""

CoordAbsJac = NewType('CoordAbsJac', np.ndarray[(N, ..., 3,2), np.dtype[np.floating]])
r"""Jacobian of the absolute coordinates w.r.t local coordinates

.. math::

  \ranktwo{J}_\rankone{r} = \nabla_{\rankone{q}} \rankone{r} =
  \begin{bmatrix}
    \frac{\partial x}{\partial q_0} & \frac{\partial x}{\partial q_1} \\
    \frac{\partial y}{\partial q_0} & \frac{\partial y}{\partial q_1} \\
    \frac{\partial z}{\partial q_0} & \frac{\partial z}{\partial q_1}
  \end{bmatrix}
"""

CoordGeom = NewType('CoordGeom', np.ndarray[(Union[N,Literal[1]], ..., 2,2,3), np.dtype[np.floating]])
r"""Absolute coordinates :math:`\in \mathbb{R}^3` at 4 vertices

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

Cells = NewType('Cells', np.ndarray[(NC, 2, 2), np.dtype[np.integer]])
r"""Mapping cell :math:`\in` :class:`~p4est.typing.NC` ⟶ vertex
:math:`\in` :class:`~p4est.typing.NV`.

Indexing is ``[cell, ∓y, ∓x]``

.. code-block::

  cells[:,0,0] ⟶ Vertex(-y, -x)
  cells[:,0,1] ⟶ Vertex(-y, +x)
  cells[:,1,0] ⟶ Vertex(+y, -x)
  cells[:,1,1] ⟶ Vertex(+y, +x)
"""

CellAdj = NewType('CellAdj', np.ndarray[(NC, 2, 2), np.dtype[np.integer]])
r"""Topological connectivity to other cells accross each face,
cell :math:`\in` :class:`~p4est.typing.NC` ⟶ *cell*
:math:`\in` :class:`~p4est.typing.NC`.

Indexing is ``[cell, (x,y), ∓(x|y)]``

.. code-block::

  cell_adj[:,0,0] ⟶ Cell(-xface)
  cell_adj[:,0,1] ⟶ Cell(+xface)
  cell_adj[:,1,0] ⟶ Cell(-yface)
  cell_adj[:,1,1] ⟶ Cell(+yface)
"""

CellAdjFace = NewType('CellAdjFace', np.ndarray[(NC, 2, 2), np.dtype[np.integer]])
r"""Topological order of the faces of each connected cell,
cell :math:`\in` :class:`~p4est.typing.NC` ⟶ *cell-face* :math:`\in [0,7]`

Indexing is ``[cell, (x,y), ∓(x|y)]``
"""

CellNodes = NewType('CellNodes', np.ndarray[(NC, 2, 2), np.dtype[np.integer]])
r"""Mapping cell :math:`\in` :class:`~p4est.typing.NC` ⟶ node
:math:`\in` :class:`~p4est.typing.NN`

Indexing is ``[cell, ∓y, ∓x]``

.. code-block::

  cell_nodes[:,0,0] ⟶ Node(-y, -x)
  cell_nodes[:,0,1] ⟶ Node(-y, +x)

  cell_nodes[:,1,0] ⟶ Node(+y, -x)
  cell_nodes[:,1,1] ⟶ Node(+y, +x)
"""

NodeCells = NewType('NodeCells', jagged_array[NN, np.ndarray[M, np.dtype[np.integer]]])
r"""Mapping node :math:`\in` :class:`~p4est.typing.NN` ⟶ cell
:math:`\in` :class:`~p4est.typing.NC`

Indexing is ``[node, cell]``
"""

NodeCellsInv = NewType('NodeCellsInv', jagged_array[NN, np.ndarray[M, np.dtype[np.integer]]])
r"""Mapping node :math:`\in` :class:`~p4est.typing.NN` ⟶ *cell-node* :math:`\in [0,3]`

Indexing is ``[node, cell]``
"""
