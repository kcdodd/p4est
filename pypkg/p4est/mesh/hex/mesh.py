import numpy as np

from .base import HexMeshBase
from ...geom import (
  interp_trilinear,
  interp_slerp_quad,
  interp_sphere_to_cart_slerp )
from .topo import (
  hex_cell_nodes,
  hex_cell_edges,
  hex_cell_adj )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexMesh(HexMeshBase):
  """Conveience constructor for hexahedral mesh

  Parameters
  ----------
  verts : numpy.ndarray
    shape = (NV, 3), dtype = np.float64

    Position of each vertex.

  cells : numpy.ndarray
    shape = (NC, 2, 2, 2), dtype = np.int32

    Mapping of hexahedral cells to the indices of their 8 vertices.
    (AKA :c:var:`p8est_connectivity_t.tree_to_vertex`)

    .. code-block::

      cells[:,0,0,0] -> vert(-z, -y, -x)
      cells[:,0,0,1] -> vert(-z, -y, +x)

      cells[:,0,1,0] -> vert(-z, +y, -x)
      cells[:,0,1,1] -> vert(-z, +y, +x)

      cells[:,1,0,0] -> vert(+z, -y, -x)
      cells[:,1,0,1] -> vert(+z, -y, +x)

      cells[:,1,1,0] -> vert(+z, +y, -x)
      cells[:,1,1,1] -> vert(+z, +y, +x)

  vert_nodes : None | numpy.ndarray
    shape = (NV,), dtype = np.int32

    The topological node associated with each vertex, causing cells to be connected
    by having vertices associated with the same node in addition to directly
    sharing vertices.
    A value of ``-1`` is used to indicate independent vertices.
    If not given, each vertex is assumed to be independent, and cells are only
    connected by shared vertices.

  """
  def __init__(self,
    verts,
    cells,
    vert_nodes = None ):

    verts = np.ascontiguousarray(
      verts,
      dtype = np.float64 ).reshape(-1, 3)

    cells = np.ascontiguousarray(
      cells,
      dtype = np.int32 ).reshape(-1, 2, 2, 2)

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
    ( self._vert_nodes,
      cell_nodes,
      node_cells,
      node_cells_inv ) = hex_cell_nodes(cells, vert_nodes)

    independent = self._vert_nodes == -1
    ni = np.count_nonzero(independent)

    full_vert_nodes = np.copy(self._vert_nodes)
    full_vert_nodes[independent] = np.arange(ni) + np.amax(self._vert_nodes) + 1

    full_cell_nodes = full_vert_nodes[cells]

    ( cell_edges,
      edge_cells,
      edge_cells_inv ) = hex_cell_edges(full_cell_nodes)

    ( cell_adj,
      cell_adj_face ) = hex_cell_adj(full_cell_nodes)

    super().__init__(
      verts = verts,
      cells = cells,
      cell_adj = cell_adj,
      cell_adj_face = cell_adj_face,
      cell_edges = cell_edges,
      edge_cells = edge_cells,
      edge_cells_inv = edge_cells_inv,
      cell_nodes = cell_nodes,
      node_cells = node_cells,
      node_cells_inv = node_cells_inv )

  #-----------------------------------------------------------------------------
  @property
  def vert_nodes(self):
    return self._vert_nodes

  #-----------------------------------------------------------------------------
  def coord(self,
    offset,
    where = None ):

    if where is None:
      where = slice(None)

    cell_verts = self.verts[ self.cells[ where ] ]

    uv = np.clip(np.asarray(offset), 0.0, 1.0)

    return interp_trilinear(cell_verts, uv)

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

    p.add_mesh_clip_plane(
      mesh = _grid,
      scalars = 'root',
      show_edges = True,
      line_width = 1,
      normal='x',
      invert = True,
      crinkle = True)

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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexMeshSpherical(HexMesh):
  #-----------------------------------------------------------------------------
  def coord(self,
    offset,
    where = None ):

    if where is None:
      where = slice(None)

    cell_verts = self.verts[ self.cells[ where ] ]

    uv = np.clip(np.asarray(offset), 0.0, 1.0)

    return interp_sphere_to_cart_slerp(cell_verts, uv)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexMeshCartesianSpherical(HexMesh):
  #-----------------------------------------------------------------------------
  def coord(self,
    offset,
    where = None ):

    if where is None:
      where = slice(None)

    cell_verts = self.verts[ self.cells[ where ] ]

    uv = np.clip(np.asarray(offset), 0.0, 1.0)

    return interp_slerp_quad(cell_verts, uv)