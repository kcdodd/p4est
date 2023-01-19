import numpy as np
import pyvista as pv
from p4est import (
  QuadAMR,
  interp_sphere_to_cart_slerp2,
  interp_slerp2)

from p4est.mesh.quad import (
  QuadMesh,
  icosahedron_golden,
  icosahedron_spherical,
  icosahedron,
  cube,
  spherical_cube)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_grid(grid):
  scale = 0.99
  _scale = 1.0 - scale

  pv.set_plot_theme('paraview')
  p = pv.Plotter()

  nc = len(grid.local)
  verts = np.empty((4*nc, 3))
  verts[:nc] = grid.coord(offset = (_scale, _scale))
  verts[nc:2*nc] = grid.coord(offset = (_scale, scale))
  verts[2*nc:3*nc] = grid.coord(offset = (scale, scale))
  verts[3*nc:] = grid.coord(offset = (scale, _scale))

  idx = np.arange(nc)

  faces = np.empty((nc, 5), dtype = np.int32)
  faces[:,0] = 4
  faces[:nc,1] = idx
  faces[:nc,2] = idx + nc
  faces[:nc,3] = idx + 2*nc
  faces[:nc,4] = idx + 3*nc

  p.add_mesh(
    pv.PolyData(verts, faces = faces.ravel()),
    scalars = grid.local.root,
    show_edges = True,
    line_width = 1,
    point_size = 3 )

  p.add_axes()
  p.add_cursor(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
  p.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_icosahedron_golden():
  mesh = icosahedron_golden()
  mesh.show()

  grid = QuadAMR(
    mesh = mesh)

  for r in range(3):
    grid.local.adapt = 1
    grid.adapt()
  plot_grid(grid)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_icosahedron_spherical():
  mesh = icosahedron_spherical()

  grid = QuadAMR(mesh = mesh)

  for r in range(3):
    grid.local.adapt = 1
    grid.adapt()
  plot_grid(grid)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_icosahedron():
  mesh = icosahedron()
  mesh.show()

  grid = QuadAMR(mesh = mesh)

  for r in range(4):
    grid.local.adapt = 1
    grid.adapt()
  plot_grid(grid)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_cube():
  mesh = cube()

  grid = QuadAMR(mesh = mesh)

  for r in range(4):
    grid.local.adapt = 1
    grid.adapt()
  plot_grid(grid)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_spherical_cube():
  mesh = spherical_cube()

  grid = QuadAMR(mesh = mesh)

  for r in range(4):
    grid.local.adapt = 1
    grid.adapt()
  plot_grid(grid)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
  run_icosahedron_golden()
  run_icosahedron_spherical()
  run_icosahedron()
  run_cube()
  run_spherical_cube()
