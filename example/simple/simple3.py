import numpy as np
import itertools
import pyvista as pv
from p4est import (
  P8est,
  interp_sphere_to_cart_slerp,
  interp_slerp_quad)

from p4est.mesh.hex import (
  HexMesh,
  cube )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_mesh(mesh):

  pv.set_plot_theme('paraview')
  p = pv.Plotter()

  nc = len(mesh.cells)

  faces = np.empty((nc, 5), dtype = mesh.cells.dtype)
  faces[:,0] = 4
  faces[:nc,1] = mesh.cells[:,0,0]
  faces[:nc,2] = mesh.cells[:,0,1]
  faces[:nc,3] = mesh.cells[:,1,1]
  faces[:nc,4] = mesh.cells[:,1,0]

  verts = mesh.verts

  p.add_mesh(
    pv.PolyData(verts, faces = faces.ravel()),
    # scalars = np.arange(len(mesh.verts)),
    scalars = np.arange(len(mesh.cells)),
    show_edges = True,
    line_width = 1,
    point_size = 3 )


  for i in range(len(mesh.node_cells)):
    m = mesh.vert_nodes == i
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
def plot_grid(grid, interp = None):
  scale = 0.99
  _scale = 1.0 - scale

  scales = [_scale, scale]

  pv.set_plot_theme('paraview')
  p = pv.Plotter()

  # nc = len(grid.leaf_info)
  # verts = np.empty((nc, 2,2,2,3))

  # for i,j,k in itertools.product([0,1], repeat=3):
  #   verts[:,i,j,k] = grid.leaf_coord(
  #     uv = (scales[i], scales[j], scales[k]),
  #     interp = interp)

  # idx = np.arange(nc)

  # cells = np.empty((nc, 9), dtype = np.int32)
  # cells[:,0] = 8
  # cells[:nc,1:] = idx[:,None] + np.arange(8)[None,:]

  points = grid.leaf_coord(
    uv = (0.5, 0.5, 0.5))


  p.add_points(
    points,
    point_size = 7,
    color = 'red',
    opacity = 0.75 )

  # p.add_mesh(
  #   pv.UnstructuredGrid(cells, [pv.CellType.HEXAHEDRON]*nc, verts.reshape(-1,3)),
  #   scalars = grid.leaf_info.root,
  #   show_edges = True,
  #   line_width = 1 )


  p.add_axes()
  p.add_cursor(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
  p.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_cube():
  mesh = cube()

  grid = P8est(
    mesh = mesh,
    min_level = 0)

  for r in range(3):
    grid.leaf_info.adapt = 1
    grid.adapt()

  plot_grid(grid)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
  run_cube()
