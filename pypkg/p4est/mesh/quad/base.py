# Enable postponed evaluation of annotations
from __future__ import annotations
from typing import (
  Optional,
  Union,
  Literal,
  TypeVar,
  NewType )
import numpy as np
from collections.abc import Sequence
from ...utils import (
  jagged_array )
from .geom import (
  QuadGeometry,
  QuadLinear )
from .topo import (
  quad_cell_nodes,
  quad_cell_adj )


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#: A variable size
N = TypeVar('N', bound = int)

#: A variable number of vertices
NV = TypeVar('NV', bound = int)

#: A variable number of cells
NC = TypeVar('NC', bound = int)

#: A variable number of nodes
NN = TypeVar('NN', bound = int)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadMesh:
  r"""Base container for quadrilateral mesh

  Parameters
  ----------
  verts:
    Position of each vertex.
    (AKA :c:var:`p4est_connectivity_t.vertices`)

  cells:
    Mapping of quadrilateral cells to the indices of their 4 vertices.
    (AKA :c:var:`p4est_connectivity_t.tree_to_vertex`)

  vert_nodes:
    The topological node associated with each vertex, causing cells to be connected
    by having vertices associated with the same node in addition to directly
    sharing vertices.
    A value of ``-1`` is used to indicate independent vertices.
    If not given, each vertex is assumed to be independent, and cells are only
    connected by shared vertices.

  geoms:
    The available geometries that may be referenced by 'vert_geom'.
    (default: [QuadLinear])

  vert_geom:
    Indices into 'geoms' to get the geometry associated with each vertex.
    (default: zeros(:class:`NV`))

  """
  def __init__(self,
    verts: np.ndarray[(NV, 2, 2), np.dtype[np.floating]],
    cells: np.ndarray[(NC, 2, 2), np.dtype[np.integer]],
    vert_nodes: Optional[np.ndarray[(NV,), np.dtype[np.integer]]] = None,
    geoms: Optional[Sequence[QuadGeometry]] = None,
    vert_geom: Optional[np.ndarray[(NV,), np.dtype[np.integer]]] = None ):

    #...........................................................................
    if vert_nodes is None:
      vert_nodes = -np.ones(len(verts), dtype = np.int32)

    vert_nodes = np.ascontiguousarray(
      vert_nodes,
      dtype = np.int32 )

    if not (
      vert_nodes.ndim == 1
      and vert_nodes.shape[0] == len(verts) ):

      raise ValueError(f"'vert_nodes' must have shape ({len(verts)},): {vert_nodes.shape}")

    _min = np.amin(vert_nodes)
    _max = np.amax(vert_nodes)

    if _min < -1 or _max >= len(verts):
      raise ValueError(f"'vert_nodes' values must be in the range [-1,{len(verts)-1}]: [{_min},{_max}]")

    #...........................................................................
    if geoms is None:
      geoms = [QuadLinear()]

    if not isinstance(geoms, Sequence):
      # raise ValueError(f"'geoms' must be list of QuadGeometry: {type(geoms)}")
      geoms = [geoms]

    if len(geoms) == 0:
      raise ValueError(f"'geoms' must have at least one QuadGeometry")

    for i, geom in enumerate(geoms):
      if not isinstance(geom, QuadGeometry):
        raise ValueError(f"'geoms' must be list of QuadGeometry: {type(geom)}")

    # TODO: remove when implementing inhomogenous geometry
    if len(geoms) != 1:
      raise NotImplementedError(f"Inhomogenous geometry not implemented")

    #...........................................................................
    if vert_geom is None:
      vert_geom = np.zeros(len(verts), dtype = np.int32)

    vert_geom = np.ascontiguousarray(
      vert_geom,
      dtype = np.int32 )

    if not (
      vert_geom.ndim == 1
      and vert_geom.shape[0] == len(verts) ):

      raise ValueError(f"'vert_geom' must have shape ({len(verts)},): {vert_geom.shape}")

    _min = np.amin(vert_geom)
    _max = np.amax(vert_geom)

    if _min < 0 or _max >= len(geoms):
      raise ValueError(f"'vert_geom' values must be in the range [0,{len(geoms)-1}]: [{_min},{_max}]")

    #...........................................................................
    ( self._vert_nodes,
      cell_nodes,
      node_cells,
      node_cells_inv ) = quad_cell_nodes(cells, vert_nodes)

    independent = self._vert_nodes == -1
    ni = np.count_nonzero(independent)

    full_vert_nodes = np.copy(self._vert_nodes)
    full_vert_nodes[independent] = np.arange(ni) + np.amax(self._vert_nodes) + 1

    ( cell_adj,
      cell_adj_face ) = quad_cell_adj(full_vert_nodes[cells])

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

    self._geoms = tuple(geoms)
    self._vert_geom = vert_geom

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
  def verts(self) \
      -> np.ndarray[(NV, 2, 2), np.dtype[np.floating]]:
    """Position of each vertex.
    (AKA :c:var:`p4est_connectivity_t.vertices`)
    """
    return self._verts

  #-----------------------------------------------------------------------------
  @property
  def vert_nodes(self) \
      -> np.ndarray[(NV,), np.dtype[np.integer]]:
    """The topological node associated with each vertex, causing cells to be connected
    by having vertices associated with the same node in addition to directly
    sharing vertices.
    """
    return self._vert_nodes

  #-----------------------------------------------------------------------------
  @property
  def cells(self) \
      -> np.ndarray[(NC, 2, 2), np.dtype[np.integer]]:
    """Mapping of quadrilateral cells to the indices of their 4 vertices.
    (AKA :c:var:`p4est_connectivity_t.tree_to_vertex`)

    .. code-block::

      cells[:,0,0] -> Vertex(-y, -x)
      cells[:,0,1] -> Vertex(-y, +x)
      cells[:,1,0] -> Vertex(+y, -x)
      cells[:,1,1] -> Vertex(+y, +x)

    """
    return self._cells

  #-----------------------------------------------------------------------------
  @property
  def cell_adj(self) \
      -> np.ndarray[(NC, 2, 2), np.dtype[np.integer]]:
    """Mapping of cells to the indices of their (up to) 4 face-adjacent neighbors.
    (AKA :c:var:`p4est_connectivity_t.tree_to_tree`)

    .. code-block::

      cell_adj[:,0,0] -> Cell(-xface)
      cell_adj[:,0,1] -> Cell(+xface)
      cell_adj[:,1,0] -> Cell(-yface)
      cell_adj[:,1,1] -> Cell(+yface)
    """
    return self._cell_adj

  #-----------------------------------------------------------------------------
  @property
  def cell_adj_face(self) \
      -> np.ndarray[(NC, 2, 2), np.dtype[np.integer]]:
    """Topological order of the faces of each connected cell.
    (AKA :c:var:`p4est_connectivity_t.tree_to_face`)
    """
    return self._cell_adj_face

  #-----------------------------------------------------------------------------
  @property
  def cell_nodes(self) \
      -> np.ndarray[(NC, 2, 2), np.dtype[np.integer]]:
    """Mapping of cells to the indices of their (up to) 4 nodes
    in ``node_cells`` and ``node_cells_inv``,
    ``-1`` used where nodes are not specified.
    (AKA :c:var:`p4est_connectivity_t.tree_to_corner`)

    .. code-block::

      cell_nodes[:,0,0] -> Node(-y, -x)
      cell_nodes[:,0,1] -> Node(-y, +x)
      cell_nodes[:,1,0] -> Node(+y, -x)
      cell_nodes[:,1,1] -> Node(+y, +x)

    """
    return self._cell_nodes

  #-----------------------------------------------------------------------------
  @property
  def node_cells(self) \
      -> jagged_array:
    """Mapping to cells sharing each node, all ``len(node_cells[i]) > 1``.
    (AKA :c:var:`p4est_connectivity_t.corner_to_tree`)

    jagged_array[(NN, ...), np.dtype[np.integer]]
    """
    return self._node_cells

  #-----------------------------------------------------------------------------
  @property
  def node_cells_inv(self) \
      -> jagged_array:
    """Mapping to the cell's local vertex {0,1,2,3} in ``cell_nodes`` which maps
    back to the node.
    (AKA :c:var:`p4est_connectivity_t.corner_to_corner`)

    .. code-block::

      nodes = np.repeat(np.arange(len(node_cells)), node_cells.row_counts)
      _nodes = cell_nodes.reshape(-1,4)[(node_cells.flat, node_cells_inv.flat)]
      valid = nodes == _nodes

    jagged_array[(NN, ...), np.dtype[np.integer]]
    """
    return self._node_cells_inv

  #-----------------------------------------------------------------------------
  @property
  def geoms(self) \
      -> Sequence[QuadGeometry]:
    """The available geometries referenced by 'vert_geom'.
    """
    return self._geoms

  #-----------------------------------------------------------------------------
  @property
  def vert_geom(self) -> np.ndarray[(NV,), np.dtype[np.integer]]:
    """Indices into 'geoms' to get the geometry associated with each vertex.
    """
    return self._vert_geom

  #-----------------------------------------------------------------------------
  def coord(self,
    offset : np.ndarray[(Union[N,Literal[1]], ..., 2), np.dtype[np.floating]],
    where : Union[None, slice, np.ndarray[..., np.dtype[Union[np.integer, bool]]]] = None ) \
      -> np.ndarray[(N, ..., Literal[3]), np.dtype[np.floating]]:
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
    offset :
      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^2` along each edge of the cell.
    where :
      Subset of cells. (default: :py:obj:`slice(None)`)

    Returns
    -------
    Global absolute coordinates
    """

    if where is None:
      where = slice(None)

    offset = np.clip(np.asarray(offset), 0.0, 1.0)
    offset = np.atleast_2d(offset)
    shape = offset.shape
    offset = offset.reshape(shape[0], int(np.prod(shape[1:-1])), shape[-1])

    idx = np.atleast_1d(np.arange(len(self.cells))[where])
    _shape = idx.shape
    idx = idx.reshape(_shape[0], int(np.prod(_shape[1:])))

    out_shape = (*_shape, 3)

    assert idx.shape[1] == offset.shape[1]
    assert idx.shape[0] == 1 or offset.shape[0] == 1 or idx.shape[0] == offset.shape[0]

    # (:,:,2,2)
    cells = self.cells[idx]
    # (:,:,2,2,3)
    cell_verts = self.verts[cells]

    cell_geoms = self.vert_geom[cells]

    if len(self.geoms) == 1:
      # homogenous geometry
      return self.geoms[0].coord(cell_verts, offset).reshape(out_shape)

    raise NotImplementedError(f"Inhomogenous geometry not implemented")

  #-----------------------------------------------------------------------------
  def show(self):
    import pyvista as pv

    pv.set_plot_theme('paraview')
    p = pv.Plotter()

    nc = len(self.cells)

    faces = np.empty((nc, 5), dtype = self.cells.dtype)
    faces[:,0] = 4
    faces[:nc,1] = self.cells[:,0,0]
    faces[:nc,2] = self.cells[:,0,1]
    faces[:nc,3] = self.cells[:,1,1]
    faces[:nc,4] = self.cells[:,1,0]

    verts = self.verts

    p.add_mesh(
      pv.PolyData(verts, faces = faces.ravel()),
      scalars = np.arange(nc),
      show_edges = True,
      line_width = 1,
      point_size = 3 )

    for i in range(len(self.node_cells)):
      m = self.vert_nodes == i
      node_verts = verts[m]

      if len(node_verts):
        p.add_points(
          node_verts,
          point_size = 7,
          color = 'red',
          opacity = 0.75 )

        p.add_point_labels(
          node_verts,
          labels = [ str(i) ]*len(node_verts),
          text_color = 'yellow',
          font_size = 30,
          fill_shape = False )

    p.add_axes()
    p.add_cursor(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    p.show()

