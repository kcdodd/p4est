import numpy as np
import itertools
import pyvista as pv
from p4est import (
  P8est,
  interp_sphere_to_cart_slerp,
  interp_slerp_quad)

from p4est.mesh.hex import (
  HexMesh,
  cube,
  icosahedron_spherical_shell,
  icosahedron_shell)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_mesh(mesh):

  pv.set_plot_theme('paraview')
  p = pv.Plotter()


  verts = mesh.verts

  nc = len(mesh.cells)
  cells = np.empty((nc, 9), dtype = np.int32)
  cells[:,0] = 8
  cells[:,1:3] = mesh.cells[:,0,0,:]
  cells[:,3:5] = mesh.cells[:,0,1,::-1]
  cells[:,5:7] = mesh.cells[:,1,0,:]
  cells[:,7:] = mesh.cells[:,1,1,::-1]

  p.add_mesh(
    pv.UnstructuredGrid(cells, [pv.CellType.HEXAHEDRON]*nc, verts.reshape(-1,3)),
    scalars = np.arange(nc),
    show_edges = True,
    line_width = 1,
    opacity = 1.0)

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

  # points = grid.leaf_coord(
  #   uv = (0.5, 0.5, 0.5))

  # p.add_points(
  #   points,
  #   point_size = 7,
  #   color = 'red',
  #   opacity = 0.75 )

  nc = len(grid.leaf_info)
  verts = np.empty((nc, 2,2,2,3))

  kji = [
    (0,0,0),
    (0,0,1),
    (0,1,1),
    (0,1,0),
    (1,0,0),
    (1,0,1),
    (1,1,1),
    (1,1,0) ]

  for k,j,i in kji:
    verts[:,k,j,i] = grid.leaf_coord(
      uv = (scales[k], scales[j], scales[i]),
      interp = interp)

  vidx = (8*np.arange(nc)[:,None] + np.arange(8)[None,:]).reshape(-1,2,2,2)

  cells = np.empty((nc, 9), dtype = np.int32)
  cells[:,0] = 8
  cells[:,1:3] = vidx[:,0,0,:]
  cells[:,3:5] = vidx[:,0,1,::-1]
  cells[:,5:7] = vidx[:,1,0,:]
  cells[:,7:] = vidx[:,1,1,::-1]

  _grid = pv.UnstructuredGrid(cells, [pv.CellType.HEXAHEDRON]*nc, verts.reshape(-1,3))
  _grid.cell_data['root'] = grid.leaf_info.root

  p.add_mesh_clip_plane(
    mesh = _grid,
    scalars = 'root',
    show_edges = True,
    line_width = 1,
    normal='x',
    invert = True,
    crinkle = True)

  p.add_axes()
  p.add_cursor(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
  p.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_cube():
  mesh = cube()

  grid = P8est(
    mesh = mesh,
    min_level = 0)

  for r in range(4):
    grid.leaf_info.adapt = 1
    grid.adapt()

  plot_grid(grid, interp = None)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_icosahedron_spherical():
  mesh = icosahedron_spherical_shell()
  # plot_mesh(mesh)

  grid = P8est(
    mesh = mesh,
    min_level = 0)

  for r in range(3):
    grid.leaf_info.adapt = 1
    grid.adapt()

  plot_grid(grid, interp = interp_sphere_to_cart_slerp)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_icosahedron():
  mesh = icosahedron_shell()
  # plot_mesh(mesh)

  grid = P8est(
    mesh = mesh,
    min_level = 0)

  for r in range(3):
    grid.leaf_info.adapt = 1
    grid.adapt()

  plot_grid(grid, interp = interp_slerp_quad)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
  run_cube()
  run_icosahedron_spherical()
  run_icosahedron()
