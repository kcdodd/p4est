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
  mesh.show()

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