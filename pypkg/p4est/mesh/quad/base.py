import numpy as np
from ...utils import (
  jagged_array )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadMeshBase:
  """Base container for quadrilateral mesh

  Parameters
  ----------
  verts : ndarray with shape = (NV, 2 or 3), dtype = np.float64
    Position of each vertex.
    (AKA :c:var:`p4est_connectivity_t.vertices`)
  cells : ndarray with shape = (NC, 2, 2), dtype = np.int32
    Mapping of quadrilateral cells to the indices of their 4 vertices.
    (AKA :c:var:`p4est_connectivity_t.tree_to_vertex`)

    .. code-block::

      cells[:,0,0] -> Vertex(-y, -x)
      cells[:,0,1] -> Vertex(-y, +x)
      cells[:,1,0] -> Vertex(+y, -x)
      cells[:,1,1] -> Vertex(+y, +x)

  cell_adj : ndarray with shape = (NC, 2, 2), dtype = np.int32
    Mapping of cells to the indices of their (up to) 4 face-adjacent neighbors.
    (AKA :c:var:`p4est_connectivity_t.tree_to_tree`)

    .. code-block::

      cell_adj[:,0,0] -> Cell(-xface)
      cell_adj[:,0,1] -> Cell(+xface)
      cell_adj[:,1,0] -> Cell(-yface)
      cell_adj[:,1,1] -> Cell(+yface)

  cell_adj_face : ndarray with shape = (NC, 2, 2), dtype = np.int8
    Topological order of the faces of each connected cell.
    (AKA :c:var:`p4est_connectivity_t.tree_to_face`)

  cell_nodes : ndarray with shape = (NC, 2, 2), dtype = np.int32
    Mapping of cells to the indices of their (up to) 4 nodes
    in ``node_cells`` and ``node_cells_inv``,
    ``-1`` used where nodes are not specified.
    (AKA :c:var:`p4est_connectivity_t.tree_to_corner`)

    .. code-block::

      cell_nodes[:,0,0] -> Node(-y, -x)
      cell_nodes[:,0,1] -> Node(-y, +x)
      cell_nodes[:,1,0] -> Node(+y, -x)
      cell_nodes[:,1,1] -> Node(+y, +x)

  node_cells : jagged_array with shape = (NN, *, 1), dtype = np.int32
    Mapping to cells sharing each node, all ``len(node_cells[i]) > 1``.
    (AKA :c:var:`p4est_connectivity_t.corner_to_tree`)
  node_cells_inv : jagged_array with shape = (NN, *, 1), dtype = np.int32
    Mapping to the cell's local vertex {0,1,2,3} in ``cell_nodes`` which maps
    back to the node.
    (AKA :c:var:`p4est_connectivity_t.corner_to_corner`)

    .. code-block::

      nodes = np.repeat(np.arange(len(node_cells)), node_cells.row_counts)
      _nodes = cell_nodes.reshape(-1,4)[(node_cells.flat, node_cells_inv.flat)]
      valid = nodes == _nodes

  """
  def __init__(self,
    verts,
    cells,
    cell_adj,
    cell_adj_face,
    cell_nodes,
    node_cells,
    node_cells_inv ):

    if not (
      isinstance(node_cells, jagged_array)
      and isinstance(node_cells_inv, jagged_array)
      and node_cells.flat.shape == node_cells_inv.flat.shape
      and (
        node_cells.row_idx is node_cells_inv.row_idx
        or np.all(node_cells.row_idx == node_cells_inv.row_idx) ) ):

      raise ValueError(f"node_cells and node_cells_inv must have the same structure")

    self._verts = np.ascontiguousarray(
      verts,
      dtype = np.float64 )

    self._cells = np.ascontiguousarray(
      cells,
      dtype = np.int32 )

    self._cell_adj = np.ascontiguousarray(
      cell_adj,
      dtype = np.int32 )

    self._cell_adj_face = np.ascontiguousarray(
      cell_adj_face,
      dtype = np.int8 )

    self._cell_nodes = np.ascontiguousarray(
      cell_nodes,
      dtype = np.int32 )

    nodes = np.repeat(np.arange(len(node_cells)), node_cells.row_counts)
    _nodes = self._cell_nodes.reshape(-1,4)[(node_cells.flat, node_cells_inv.flat)]

    if not np.all(_nodes == nodes):
      raise ValueError(
        f"'node_cells' and 'node_cells_inv' are not consistent with 'node_cells'")

    # TODO: ensure proper dtype of the jagged_array?
    self._node_cells = node_cells
    self._node_cells_inv = node_cells_inv

  #-----------------------------------------------------------------------------
  @property
  def verts(self):
    return self._verts

  #-----------------------------------------------------------------------------
  @property
  def cells(self):
    return self._cells

  #-----------------------------------------------------------------------------
  @property
  def cell_adj(self):
    return self._cell_adj

  #-----------------------------------------------------------------------------
  @property
  def cell_adj_face(self):
    return self._cell_adj_face

  #-----------------------------------------------------------------------------
  @property
  def cell_nodes(self):
    return self._cell_nodes

  #-----------------------------------------------------------------------------
  @property
  def node_cells(self):
    return self._node_cells

  #-----------------------------------------------------------------------------
  @property
  def node_cells_inv(self):
    return self._node_cells_inv


  #-----------------------------------------------------------------------------
  def coord(self,
    offset,
    where = None ):
    r"""Transform to (physical/global) coordinates of a point relative to each cell

    .. math::

      \func{\rankone{r}}{\rankone{q}} =
      \begin{bmatrix}
        \func{\rankzero{x}}{\rankzero{q}_0, \rankzero{q}_1} \\
        \func{\rankzero{y}}{\rankzero{q}_0, \rankzero{q}_1} \\
        \func{\rankzero{z}}{\rankzero{q}_0, \rankzero{q}_1}
      \end{bmatrix}

    Parameters
    ----------
    offset : numpy.ndarray
      shape = (*,2)

      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^2` along each edge of the cell.
    where : None | slice | numpy.ndarray
      Subset of cells. (default: slice(None))


    Returns
    -------
    coord: array of shape = (NC, 3)
    """
    raise NotImplementedError

  #-----------------------------------------------------------------------------
  def coord_jac(self,
    offset,
    where = None ):
    r"""Jacobian of the absolute coordinates w.r.t local coordinates

    .. math::

      \ranktwo{J}_\rankone{r} = \nabla_{\rankone{q}} \rankone{r} =
      \begin{bmatrix}
        \frac{\partial x}{\partial q_0} & \frac{\partial x}{\partial q_1} \\
        \frac{\partial y}{\partial q_0} & \frac{\partial y}{\partial q_1} \\
        \frac{\partial z}{\partial q_0} & \frac{\partial z}{\partial q_1}
      \end{bmatrix}

    Parameters
    ----------
    offset : numpy.ndarray
      shape = (*,2)

      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^2` along each edge of the cell.
    where : None | slice | numpy.ndarray
      Subset of cells. (default: slice(None))

    Returns
    -------
    coord_jac: numpy.ndarray
      shape = (NC, 3, 2)
    """
    raise NotImplementedError