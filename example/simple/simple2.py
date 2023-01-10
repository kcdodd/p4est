import numpy as np
import pyvista as pv
from p4est import (
  P4est,
  interp_sphere_to_cart_slerp,
  interp_slerp_quad)

from p4est.mesh.quad import (
  QuadMesh,
  icosahedron_golden,
  icosahedron_spherical,
  icosahedron,
  cube)

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

  pv.set_plot_theme('paraview')
  p = pv.Plotter()

  nc = len(grid.leaf_info)
  verts = np.empty((4*nc, 3))
  verts[:nc] = grid.leaf_coord(uv = (_scale, _scale), interp = interp)
  verts[nc:2*nc] = grid.leaf_coord(uv = (_scale, scale), interp = interp)
  verts[2*nc:3*nc] = grid.leaf_coord(uv = (scale, scale), interp = interp)
  verts[3*nc:] = grid.leaf_coord(uv = (scale, _scale), interp = interp)

  idx = np.arange(nc)

  faces = np.empty((nc, 5), dtype = np.int32)
  faces[:,0] = 4
  faces[:nc,1] = idx
  faces[:nc,2] = idx + nc
  faces[:nc,3] = idx + 2*nc
  faces[:nc,4] = idx + 3*nc

  p.add_mesh(
    pv.PolyData(verts, faces = faces.ravel()),
    scalars = grid.leaf_info.root,
    show_edges = True,
    line_width = 1,
    point_size = 3 )


  p.add_axes()
  p.add_cursor(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
  p.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_icosahedron_golden():
  mesh = icosahedron_golden()

  grid = P4est(
    mesh = mesh,
    min_level = 0)

  for r in range(3):
    grid.leaf_info.adapt = 1
    grid.adapt()

  plot_grid(grid)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_icosahedron_spherical():
  mesh = icosahedron_spherical()

  grid = P4est(
    mesh = mesh,
    min_level = 0)

  for r in range(3):
    grid.leaf_info.adapt = 1
    grid.adapt()

  plot_grid(grid, interp = interp_sphere_to_cart_slerp)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_icosahedron():
  mesh = icosahedron()
  plot_mesh(mesh)

  grid = P4est(
    mesh = mesh,
    min_level = 0)

  for r in range(4):
    grid.leaf_info.adapt = 1
    grid.adapt()

  plot_grid(grid, interp = interp_slerp_quad)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_cube():
  mesh = cube()

  grid = P4est(
    mesh = mesh,
    min_level = 0)

  for r in range(4):
    grid.leaf_info.adapt = 1
    grid.adapt()

  plot_grid(grid, interp = interp_slerp_quad)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
  run_icosahedron_golden()
  run_icosahedron_spherical()
  run_icosahedron()
  run_cube()