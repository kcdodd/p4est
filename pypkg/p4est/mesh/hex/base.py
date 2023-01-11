import numpy as np
from p4est.utils import (
  jagged_array )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexMeshBase:
  """Base container for hexahedral mesh

  Parameters
  ----------
  verts : numpy.ndarray
    shape = (NV, 3), dtype = numpy.float64

    Position of each vertex.
    (AKA :c:var:`p8est_connectivity_t.vertices`)
    Indexing is ``[vertex, (x,y,z)]``

  cells : numpy.ndarray
    shape = (NC, 2, 2, 2), dtype = numpy.int32

    Mapping of hexahedral cells to the indices of their 8 vertices.
    (AKA :c:var:`p8est_connectivity_t.tree_to_vertex`)

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

  cell_adj : numpy.ndarray
    shape = (NC, 3, 2), dtype = numpy.int32

    Mapping of cells to the indices of their (up to) 6 face-adjacent neighbors.
    (AKA :c:var:`p8est_connectivity_t.tree_to_tree`)

    Indexing is ``[cell, (x,y,z), ∓(x|y|z)]``

    .. code-block::

      cell_adj[:,0,0] -> xface(-x)
      cell_adj[:,0,1] -> xface(+x)

      cell_adj[:,1,0] -> yface(-y)
      cell_adj[:,1,1] -> yface(+y)

      cell_adj[:,2,0] -> zface(-z)
      cell_adj[:,2,1] -> zface(+z)

  cell_adj_face : numpy.ndarray
    shape = (NC, 3, 2), dtype = numpy.int8

    Topological order of the faces of each connected cell.
    (AKA :c:var:`p8est_connectivity_t.tree_to_face`)

    Indexing is ``[cell, (x,y,z), ∓(x|y|z)]``

  cell_edges : numpy.ndarray
    shape = (NC, 3, 2, 2), dtype = numpy.int32

    Mapping of cells to the indices of their (up to) 12 edges
    in ``edge_cells`` and ``edge_cells_``,
    (AKA :c:var:`p8est_connectivity_t.tree_to_edge`)

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

  edge_cells : jagged_array
    shape = (NE, *), dtype = numpy.int32

    Mapping to cells sharing each edge, all ``len(edge_cells[i]) > 1``.
    (AKA :c:var:`p8est_connectivity_t.edge_to_tree`)

    Indexing is ``[edge, cell]``

  edge_cells_inv : jagged_array
    shape = (NE, *), dtype = numpy.int32

    Mapping to the cell's local edge {0,...11} in ``cell_edges``  which maps
    back to the edge.
    (AKA :c:var:`p8est_connectivity_t.edge_to_edge`)

    Indexing is ``[edge, cell]``

    .. code-block::

      edges = np.repeat(np.arange(len(edge_cells)), edge_cells.row_counts)
      _edges = cell_edges.reshape(-1,12)[(edge_cells.flat, edge_cells_inv.flat)]
      assert np.all(edges == _edges)

  cell_nodes : numpy.ndarray
    shape = (NC, 2, 2, 2), dtype = numpy.int32

    Mapping of cells to the indices of their (up to) 8 nodes
    in ``node_cells`` and ``node_cells_inv``,
    ``-1`` used where nodes are not specified.
    (AKA :c:var:`p8est_connectivity_t.tree_to_corner`)

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

  node_cells : jagged_array
    shape = (NN, *), dtype = numpy.int32

    Mapping to cells sharing each node, all ``len(node_cells[i]) > 1``.
    (AKA :c:var:`p8est_connectivity_t.corner_to_tree`)

    Indexing is ``[node, cell]``

  node_cells_inv : jagged_array
    shape = (NN, *), dtype = numpy.int32

    Mapping to the cell's local vertex {0,...7} in ``cell_nodes`` which maps
    back to the node.
    (AKA :c:var:`p8est_connectivity_t.corner_to_corner`)

    Indexing is ``[node, cell]``

    .. code-block::

      nodes = np.repeat(np.arange(len(node_cells)), node_cells.row_counts)
      _nodes = cell_nodes.reshape(-1,8)[(node_cells.flat, node_cells_inv.flat)]
      assert np.all(nodes == _nodes)

  """
  def __init__(self,
    verts,
    cells,
    cell_adj,
    cell_adj_face,
    cell_edges,
    edge_cells,
    edge_cells_inv,
    cell_nodes,
    node_cells,
    node_cells_inv ):

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

    self._cell_edges = np.ascontiguousarray(
      cell_edges,
      dtype = np.int32 )

    self._cell_nodes = np.ascontiguousarray(
      cell_nodes,
      dtype = np.int32 )

    #...........................................................................
    # TODO: ensure proper dtype of the jagged_array?
    if not (
      isinstance(edge_cells, jagged_array)
      and isinstance(edge_cells_inv, jagged_array)
      and edge_cells.flat.shape == edge_cells_inv.flat.shape
      and (
        edge_cells.row_idx is edge_cells_inv.row_idx
        or np.all(edge_cells.row_idx == edge_cells_inv.row_idx) ) ):

      raise ValueError(f"edge_cells and edge_cells_inv must have the same structure")

    edges = np.repeat(np.arange(len(edge_cells)), edge_cells.row_counts)
    _edges = cell_edges.reshape(-1,12)[(edge_cells.flat, edge_cells_inv.flat)]

    if not np.all(_edges == edges):
      raise ValueError(
        f"'cell_edges' is not consistent with 'edge_cells' and 'edge_cells_inv'")

    #...........................................................................
    if not (
      isinstance(node_cells, jagged_array)
      and isinstance(node_cells_inv, jagged_array)
      and node_cells.flat.shape == node_cells_inv.flat.shape
      and (
        node_cells.row_idx is node_cells_inv.row_idx
        or np.all(node_cells.row_idx == node_cells_inv.row_idx) ) ):

      raise ValueError(f"node_cells and node_cells_inv must have the same structure")


    nodes = np.repeat(np.arange(len(node_cells)), node_cells.row_counts)
    _nodes = self._cell_nodes.reshape(-1,8)[(node_cells.flat, node_cells_inv.flat)]

    if not np.all(_nodes == nodes):
      raise ValueError(
        f"'cell_nodes' is not consistent with 'node_cells' and 'node_cells_inv'.")

    #...........................................................................
    self._edge_cells = edge_cells
    self._edge_cells_inv = edge_cells_inv

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
  def cell_edges(self):
    return self._cell_edges

  #-----------------------------------------------------------------------------
  @property
  def edge_cells(self):
    return self._edge_cells

  #-----------------------------------------------------------------------------
  @property
  def edge_cells_inv(self):
    return self._edge_cells_inv

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
        \func{\rankzero{x}}{\rankzero{q}_0, \rankzero{q}_1, \rankzero{q}_2} \\
        \func{\rankzero{y}}{\rankzero{q}_0, \rankzero{q}_1, \rankzero{q}_2} \\
        \func{\rankzero{z}}{\rankzero{q}_0, \rankzero{q}_1, \rankzero{q}_2}
      \end{bmatrix}

    Parameters
    ----------
    offset : numpy.ndarray
      shape = (*,3)

      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^3` along each edge of the cell.
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
        \frac{\partial x}{\partial q_0} & \frac{\partial x}{\partial q_1} & \frac{\partial x}{\partial q_2} \\
        \frac{\partial y}{\partial q_0} & \frac{\partial y}{\partial q_1} & \frac{\partial y}{\partial q_2} \\
        \frac{\partial z}{\partial q_0} & \frac{\partial z}{\partial q_1} & \frac{\partial z}{\partial q_2}
      \end{bmatrix}

    Parameters
    ----------
    offset : numpy.ndarray
      shape = (*,3)

      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^3` along each edge of the cell.
    where : None | slice | numpy.ndarray
      Subset of cells. (default: slice(None))

    Returns
    -------
    coord_jac: numpy.ndarray
      shape = (NC, 3, 3)
    """
    raise NotImplementedError