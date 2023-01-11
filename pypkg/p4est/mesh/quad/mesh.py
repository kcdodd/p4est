import numpy as np
from ...utils import (
  jagged_array )
from ...geom import (
  interp_bilinear,
  interp_slerp_quad,
  interp_sphere_to_cart_slerp )
from .base import QuadMeshBase
from .topo import (
  quad_cell_nodes,
  quad_cell_adj )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadMesh(QuadMeshBase):
  """Conveience constructor for quadrilateral mesh

  Parameters
  ----------
  verts : ndarray with shape = (NV, 2 or 3), dtype = np.float64
    Position of each vertex.
  cells : ndarray with shape = (NC, 2, 2), dtype = np.int32
    Quadrilateral cells, each defined by the indices of its 4 vertices.

    .. note::

      The order of the vertices must follow the ordering [[V00, V01], [V10, V11]],
      or [[V0, V1], [V2, V3]], for the desired geometry of the cell.
      E.G For an aligned rectangle: [[lower-left, lower-right], [upper-left,
      upper-right]].

  vert_nodes : None | ndarray with shape = (NV,), dtype = np.int32
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
      node_cells_inv ) = quad_cell_nodes(cells, vert_nodes)

    independent = self._vert_nodes == -1
    ni = np.count_nonzero(independent)

    full_vert_nodes = np.copy(self._vert_nodes)
    full_vert_nodes[independent] = np.arange(ni) + np.amax(self._vert_nodes) + 1

    ( cell_adj,
      cell_adj_face ) = quad_cell_adj(full_vert_nodes[cells])

    super().__init__(
      verts = verts,
      cells = cells,
      cell_adj = cell_adj,
      cell_adj_face = cell_adj_face,
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

    return interp_bilinear(cell_verts, uv)

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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadMeshSpherical(QuadMesh):
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
class QuadMeshCartesianSpherical(QuadMesh):
  #-----------------------------------------------------------------------------
  def coord(self,
    offset,
    where = None ):

    if where is None:
      where = slice(None)

    cell_verts = self.verts[ self.cells[ where ] ]

    uv = np.clip(np.asarray(offset), 0.0, 1.0)

    return interp_slerp_quad(cell_verts, uv)