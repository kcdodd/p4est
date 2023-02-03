# Enable postponed evaluation of annotations
from __future__ import annotations
from partis.utils import TYPING

if TYPING:
  from typing import (
    Union,
    Literal )
  from ...typing import N, M, NV, NN, NE, NC, Where
  from .typing import (
    Vertices,
    VertGeom,
    VertNodes,
    Cells,
    CellAdj,
    CellAdjFace,
    CellNodes,
    NodeCells,
    NodeCellsInv,
    NodeEdgeCounts,
    CellEdges,
    EdgeCells,
    EdgeCellsInv,
    EdgeCellCounts,
    CoordRel,
    CoordAbs)

from copy import copy
from collections.abc import Sequence
import numpy as np

from ...utils import (
  jagged_array )
from .geom import (
  HexGeometry,
  HexLinear )
from .topo import (
  hex_cell_nodes,
  hex_cell_edges,
  hex_cell_adj )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexMesh:
  r"""Base container for hexahedral mesh

  Parameters
  ----------
  verts :
    Position of each vertex.
    (AKA :c:var:`p8est_connectivity_t.vertices`)
  cells :
    Mapping of hexahedral cells to the indices of their 8 vertices.
    (AKA :c:var:`p8est_connectivity_t.tree_to_vertex`)
  vert_nodes :
    The topological node associated with each vertex, causing cells to be connected
    by having vertices associated with the same node in addition to directly
    sharing vertices.
    A value of ``-1`` is used to indicate independent vertices.
    If not given, each vertex is assumed to be independent, and cells are only
    connected by shared vertices.
  geoms :
    The available geometries that may be referenced by 'vert_geom'.
    (default: [:class:`~p4est.mesh.hex.geom.HexLinear`])
  vert_geom :
    Indices into 'geoms' to get the geometry associated with each vertex.
    (default: zeros(:class:`~p4est.typing.NV`))

  Notes
  -----

  .. figure:: ../img/topology.svg.png
    :width: 90%
    :align: center

    Topological elements.
  """
  def __init__(self,
    verts : Vertices,
    cells : Cells,
    vert_nodes : VertNodes = None,
    geoms : Sequence[HexGeometry] = None,
    vert_geom : VertGeom = None ):

    verts = np.ascontiguousarray(
      verts,
      dtype = np.float64 )

    if verts.ndim != 2 or verts.shape[1] != 3:
      raise ValueError(f"'verts' must have shape (NV,3): {verts.ndim}")

    #...........................................................................
    cells = np.ascontiguousarray(
      cells,
      dtype = np.int32 )

    if cells.ndim != 4 or cells.shape[1:] != (2,2,2):
      raise ValueError(f"'cells' must have shape (NC,2,2,2): {cells.ndim}")

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
      geoms = [HexLinear()]

    if not isinstance(geoms, Sequence):
      # raise ValueError(f"'geoms' must be list of HexGeometry: {type(geoms)}")
      geoms = [geoms]

    if len(geoms) == 0:
      raise ValueError(f"'geoms' must have at least one HexGeometry")

    for i, geom in enumerate(geoms):
      if not isinstance(geom, HexGeometry):
        raise ValueError(f"'geoms' must be list of HexGeometry: {type(geom)}")

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
      node_cells_inv ) = hex_cell_nodes(cells, vert_nodes)


    cell_nodes = self._vert_nodes[cells]

    ( cell_edges,
      edge_cells,
      edge_cells_inv,
      self._edge_cell_counts,
      self._node_edge_counts ) = hex_cell_edges(cell_nodes)

    ( cell_adj,
      cell_adj_face ) = hex_cell_adj(cell_nodes)


    self._verts = np.ascontiguousarray(
      verts,
      dtype = np.float64 )

    self._cells = np.ascontiguousarray(
      cells,
      dtype = np.int32 )

    self._cell_adj = cell_adj

    self._cell_adj_face = cell_adj_face

    self._cell_edges = cell_edges

    self._cell_nodes = cell_nodes

    self._geoms = tuple(geoms)
    self._vert_geom = vert_geom

    #...........................................................................
    if not (
      isinstance(node_cells, jagged_array)
      and isinstance(node_cells_inv, jagged_array)
      and node_cells.flat.shape[0] == node_cells_inv.flat.shape[0]
      and (
        node_cells.row_idx is node_cells_inv.row_idx
        or np.all(node_cells.row_idx == node_cells_inv.row_idx) ) ):

      raise ValueError(f"node_cells and node_cells_inv must have the same structure")


    nodes = np.repeat(np.arange(len(node_cells)), node_cells.row_counts)
    _nodes = self._cell_nodes[
      (node_cells.flat,
       node_cells_inv.flat[:,0],
       node_cells_inv.flat[:,1],
       node_cells_inv.flat[:,2])]

    if not np.all(_nodes == nodes):
      raise ValueError(
        f"'cell_nodes' is not consistent with 'node_cells' and 'node_cells_inv'.")

    #...........................................................................
    # TODO: ensure proper dtype of the jagged_array?
    if not (
      isinstance(edge_cells, jagged_array)
      and isinstance(edge_cells_inv, jagged_array)
      and edge_cells.flat.shape[0] == edge_cells_inv.flat.shape[0]
      and (
        edge_cells.row_idx is edge_cells_inv.row_idx
        or np.all(edge_cells.row_idx == edge_cells_inv.row_idx) ) ):

      raise ValueError(f"edge_cells and edge_cells_inv must have the same structure")

    edges = np.repeat(np.arange(len(edge_cells)), edge_cells.row_counts)
    _edges = cell_edges[(
      edge_cells.flat,
      edge_cells_inv.flat[:,0],
      edge_cells_inv.flat[:,1],
      edge_cells_inv.flat[:,2])]

    if not np.all(_edges == edges):
      raise ValueError(
        f"'cell_edges' is not consistent with 'edge_cells' and 'edge_cells_inv'")

    #...........................................................................
    self._edge_cells = edge_cells
    self._edge_cells_inv = edge_cells_inv

    self._node_cells = node_cells
    self._node_cells_inv = node_cells_inv

  #-----------------------------------------------------------------------------
  def __class_getitem__(cls, *args):
    from types import GenericAlias
    return GenericAlias(cls, args)

  #-----------------------------------------------------------------------------
  def __copy__(self):
    cls = type(self)
    mesh = cls.__new__(cls)

    mesh._verts = copy(self._verts)
    mesh._cells = copy(self._cells)
    mesh._cell_adj = copy(self._cell_adj)
    mesh._cell_adj_face = copy(self._cell_adj_face)
    mesh._cell_edges = copy(self._cell_edges)
    mesh._cell_nodes = copy(self._cell_nodes)
    mesh._edge_cells = copy(self._edge_cells)
    mesh._edge_cells_inv = copy(self._edge_cells_inv)
    mesh._node_cells = copy(self._node_cells)
    mesh._node_cells_inv = copy(self._node_cells_inv)
    mesh._vert_nodes = copy(self._vert_nodes)
    mesh._edge_cell_counts = copy(self._edge_cell_counts)
    mesh._node_edge_counts = copy(self._node_edge_counts)

    mesh._geoms = copy(self._geoms)
    mesh._vert_geom = copy(self._vert_geom)

    return mesh

  #-----------------------------------------------------------------------------
  @property
  def verts(self) -> Vertices:
    """Position of each vertex.
    (AKA :c:var:`p8est_connectivity_t.vertices`)
    Indexing is ``[vertex, (x,y,z)]``
    """
    return self._verts

 #-----------------------------------------------------------------------------
  @property
  def vert_nodes(self) -> VertNodes:
    """The topological node associated with each vertex, causing cells to be connected
    by having vertices associated with the same node in addition to directly
    sharing vertices.
    """
    return self._vert_nodes

  #-----------------------------------------------------------------------------
  @property
  def cells(self) -> Cells:
    """Mapping of hexahedral cells to the indices of their 8 vertices.
    (AKA :c:var:`p8est_connectivity_t.tree_to_vertex`)
    """
    return self._cells

  #-----------------------------------------------------------------------------
  @property
  def cell_adj(self) -> CellAdj:
    """Mapping of cells to the indices of their (up to) 6 face-adjacent neighbors.
    (AKA :c:var:`p8est_connectivity_t.tree_to_tree`)
    """
    return self._cell_adj

  #-----------------------------------------------------------------------------
  @property
  def cell_adj_face(self) -> CellAdjFace:
    """Topological order of the faces of each connected cell.
    (AKA :c:var:`p8est_connectivity_t.tree_to_face`)
    """
    return self._cell_adj_face

  #-----------------------------------------------------------------------------
  @property
  def cell_edges(self) -> CellEdges:
    """Mapping of cells to the indices of their (up to) 12 edges
    in ``edge_cells`` and ``edge_cells_``,
    (AKA :c:var:`p8est_connectivity_t.tree_to_edge`)
    """
    return self._cell_edges

  #-----------------------------------------------------------------------------
  @property
  def edge_cells(self) -> EdgeCells:
    """Mapping to cells sharing each edge, all ``len(edge_cells[i]) > 1``.
    (AKA :c:var:`p8est_connectivity_t.edge_to_tree`)
    """
    return self._edge_cells

  #-----------------------------------------------------------------------------
  @property
  def edge_cells_inv(self) -> EdgeCellsInv:
    """Mapping to the cell's local edge {0,...11} in ``cell_edges``  which maps
    back to the edge.
    (AKA :c:var:`p8est_connectivity_t.edge_to_edge`)

    .. code-block::

      edges = np.repeat(np.arange(len(edge_cells)), edge_cells.row_counts)
      _edges = cell_edges.reshape(-1,12)[(edge_cells.flat, edge_cells_inv.flat)]
      assert np.all(edges == _edges)
    """
    return self._edge_cells_inv

  #-----------------------------------------------------------------------------
  @property
  def cell_nodes(self) -> Cells:
    """Mapping of cells to the indices of their (up to) 8 nodes
    in ``node_cells`` and ``node_cells_inv``,
    ``-1`` used where nodes are not specified.
    (AKA :c:var:`p8est_connectivity_t.tree_to_corner`)
    """
    return self._cell_nodes

  #-----------------------------------------------------------------------------
  @property
  def node_cells(self) -> NodeCells:
    """Mapping to cells sharing each node, all ``len(node_cells[i]) > 1``.
    (AKA :c:var:`p8est_connectivity_t.corner_to_tree`)
    """
    return self._node_cells

  #-----------------------------------------------------------------------------
  @property
  def node_cells_inv(self) -> NodeCellsInv:
    """Mapping to the cell's local vertex {0,...7} in ``cell_nodes`` which maps
    back to the node.
    (AKA :c:var:`p8est_connectivity_t.corner_to_corner`)

    .. code-block::

      nodes = np.repeat(np.arange(len(node_cells)), node_cells.row_counts)
      _nodes = cell_nodes.reshape(-1,8)[(node_cells.flat, node_cells_inv.flat)]
      assert np.all(nodes == _nodes)
    """
    return self._node_cells_inv

  #-----------------------------------------------------------------------------
  @property
  def edge_cell_counts(self) -> EdgeCellCounts:
    """
    """
    return self._edge_cell_counts

  #-----------------------------------------------------------------------------
  @property
  def node_edge_counts(self) -> NodeEdgeCounts:
    """
    """
    return self._node_edge_counts

  #-----------------------------------------------------------------------------
  @property
  def geoms(self) -> Sequence[HexGeometry]:
    """The available geometries referenced by 'vert_geom'.
    """
    return self._geoms

  #-----------------------------------------------------------------------------
  @property
  def vert_geom(self) -> VertGeom:
    """Indices into 'geoms' to get the geometry associated with each vertex.
    """
    return self._vert_geom

  #-----------------------------------------------------------------------------
  def coord(self,
    offset : CoordRel,
    where : Where = None ) -> CoordAbs:
    r"""Transform to (physical/global) coordinates of a point relative to each cell

    Parameters
    ----------
    offset :
      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^3` along each edge of the cell.
      ``N = len(cells[where])``
    where :
      Subset of cells. (default: slice(None))


    Returns
    -------
    Absolute coordinates at each ``offset``
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

    assert idx.shape[1] == offset.shape[1]
    assert idx.shape[0] == 1 or offset.shape[0] == 1 or idx.shape[0] == offset.shape[0]

    # (:,:,2,2,2)
    cells = self.cells[idx]
    # (:,:,2,2,2,3)
    cell_verts = self.verts[cells]

    cell_geoms = self.vert_geom[cells]

    if len(self.geoms) == 1:
      # homogenous geometry
      return self.geoms[0].coord(cell_verts, offset)

    # determined by multiple geometries on surface of each cell
    # TODO: mixing geometries not quite working in general
    coord = np.zeros((len(cell_verts), *offset.shape[1:]), dtype = np.float64)

    # tri-linear interpolation of geometries within mixed cells
    # TODO: optimize for cells that are *not* mixed?
    b = offset
    a = 1.0 - offset
    w = np.empty(cells.shape)
    w[...,0,0,0] = a[...,2]*a[...,1]*a[...,0]
    w[...,0,0,1] = a[...,2]*a[...,1]*b[...,0]
    w[...,0,1,0] = a[...,2]*b[...,1]*a[...,0]
    w[...,0,1,1] = a[...,2]*b[...,1]*b[...,0]

    w[...,1,0,0] = b[...,2]*a[...,1]*a[...,0]
    w[...,1,0,1] = b[...,2]*a[...,1]*b[...,0]
    w[...,1,1,0] = b[...,2]*b[...,1]*a[...,0]
    w[...,1,1,1] = b[...,2]*b[...,1]*b[...,0]

    axes = tuple(range(1, cell_geoms.ndim))

    for i, geom in enumerate(self.geoms):
      # mask of cells that have at least one vertex/surface of given geometry
      m = cell_geoms == i
      _m = np.any(m, axis = (2,3,4))

      _offset = offset if offset.shape[0] == 1 else offset[_m]
      _verts = cell_verts[_m]

      v = geom.coord(_verts, _offset)

      # weight coordinates by how much each geometry participates
      _w = np.sum(w[_m] * m[_m].astype(np.float64), axis = (1,2,3))

      coord[_m] += _w[...,None] * v

    return coord

  #-----------------------------------------------------------------------------
  def show(self):
    import pyvista as pv

    pv.set_plot_theme('paraview')
    p = pv.Plotter()

    verts = self.verts

    nc = len(self.cells)
    cells = np.empty((nc, 9), dtype = np.int32)
    cells[:,0] = 8
    cells[:,1:3] = self.cells[:,0,0,:]
    cells[:,3:5] = self.cells[:,0,1,::-1]
    cells[:,5:7] = self.cells[:,1,0,:]
    cells[:,7:] = self.cells[:,1,1,::-1]

    _grid = pv.UnstructuredGrid(cells, [pv.CellType.HEXAHEDRON]*nc, verts.reshape(-1,3))
    _grid.cell_data['root'] = np.arange(nc)

    # p.add_mesh_clip_plane(
    #   mesh = _grid,
    #   # scalars = 'root',
    #   show_edges = True,
    #   line_width = 1,
    #   normal='x',
    #   invert = True,
    #   crinkle = True,
    #   show_scalar_bar = False)

    edge_verts = np.unique(
      np.sort(
        np.stack(
          ( # x-edges
            self.cells[:,0,0,:],
            self.cells[:,0,1,:],
            self.cells[:,1,0,:],
            self.cells[:,1,1,:],
            # y-edges
            self.cells[:,0,:,0],
            self.cells[:,0,:,1],
            self.cells[:,1,:,0],
            self.cells[:,1,:,1],
            # z-edges
            self.cells[:,:,0,0],
            self.cells[:,:,0,1],
            self.cells[:,:,1,0],
            self.cells[:,:,1,1] ),
          axis = 1 ).reshape(12*nc, 2),
        axis = 1 ),
      axis = 0 )

    lines = np.empty((len(edge_verts), 3), dtype = np.int32)
    lines[:,0] = 2
    lines[:,1:] = edge_verts

    edge_nodes = self.vert_nodes[edge_verts]

    _, edges = np.unique(
      np.sort(
        edge_nodes,
        axis = 1 ),
      return_inverse = True,
      axis = 0 )

    _edges = pv.PolyData(self.verts, lines = lines)
    _edges.cell_data['edge_cell_counts'] = self.edge_cell_counts[edges] - 4

    p.add_mesh(
      _edges,
      scalars = 'edge_cell_counts',
      show_edges = True,
      line_width = 3,
      opacity = 0.75,
      clim = [-4, 4],
      cmap = 'Set1' )

    _nodes = pv.PolyData(verts)
    _nodes.point_data['node_edge_counts'] = self.node_edge_counts[self.vert_nodes] - 6

    p.add_mesh(
      _nodes,
      scalars = 'node_edge_counts',
      render_points_as_spheres = True,
      point_size = 10,
      opacity = 1.0,
      clim = [-6, 6],
      cmap = 'Set2')

    p.add_axes()
    # p.add_cursor(
    #   bounds = np.array([
    #     np.amin(self.verts, axis = 0),
    #     np.amax(self.verts, axis = 0)]).T.ravel() )

    p.show()