import numpy as np
import itertools
import pyvista as pv
from p4est import (
  P8est)

from p4est.mesh.hex import (
  HexMesh,
  cube,
  spherical_cube_shell,
  spherical_cube,
  slab_spherical_cube_hole,
  icosahedron_spherical_shell,
  icosahedron_shell)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_grid(grid):
  scale = 0.99
  _scale = 1.0 - scale

  scales = [_scale, scale]

  pv.set_plot_theme('paraview')
  p = pv.Plotter()

  # points = grid.coord(
  #   uv = (0.5, 0.5, 0.5))

  # p.add_points(
  #   points,
  #   point_size = 7,
  #   color = 'red',
  #   opacity = 0.75 )

  nc = len(grid.local)
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
    verts[:,k,j,i] = grid.coord(
      offset = (scales[k], scales[j], scales[i]) )

  vidx = (8*np.arange(nc)[:,None] + np.arange(8)[None,:]).reshape(-1,2,2,2)

  cells = np.empty((nc, 9), dtype = np.int32)
  cells[:,0] = 8
  cells[:,1:3] = vidx[:,0,0,:]
  cells[:,3:5] = vidx[:,0,1,::-1]
  cells[:,5:7] = vidx[:,1,0,:]
  cells[:,7:] = vidx[:,1,1,::-1]

  _grid = pv.UnstructuredGrid(cells, [pv.CellType.HEXAHEDRON]*nc, verts.reshape(-1,3))
  _grid.cell_data['root'] = grid.local.root

  p.add_mesh_clip_plane(
    mesh = _grid,
    scalars = 'root',
    show_edges = True,
    line_width = 1,
    normal='x',
    invert = True,
    crinkle = True)

  p.add_axes()
  # p.add_cursor(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
  p.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_cube():
  mesh = cube()
  # mesh.show()

  grid = P8est(
    mesh = mesh,
    min_level = 0)

  for r in range(4):
    grid.local.adapt = 1
    grid.adapt()

  plot_grid(grid)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_spherical_cube_shell():
  mesh = spherical_cube_shell()
  # mesh.show()

  grid = P8est(
    mesh = mesh,
    min_level = 0)

  for r in range(4):
    grid.local.adapt = 1
    grid.adapt()

  plot_grid(grid)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_spherical_cube():
  mesh = spherical_cube()
  # mesh.show()

  grid = P8est(
    mesh = mesh,
    min_level = 0)

  for r in range(4):
    grid.local.adapt = 1
    grid.adapt()

  plot_grid(grid)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_slab_spherical_cube_hole():
  mesh = slab_spherical_cube_hole()
  # mesh.show()

  grid = P8est(
    mesh = mesh,
    min_level = 0)

  for r in range(4):
    grid.local.adapt = 1
    grid.adapt()

  plot_grid(grid)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_icosahedron_spherical():
  mesh = icosahedron_spherical_shell()
  # mesh.show()

  grid = P8est(
    mesh = mesh,
    min_level = 0)

  for r in range(3):
    grid.local.adapt = 1
    grid.adapt()

  plot_grid(grid)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_icosahedron():
  mesh = icosahedron_shell()
  # mesh.show()

  grid = P8est(
    mesh = mesh,
    min_level = 0)

  for r in range(3):
    grid.local.adapt = 1
    grid.adapt()

  plot_grid(grid)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
  run_cube()
  run_spherical_cube_shell()

  # NOTE: mixing geometries not quite working
  # run_slab_spherical_cube_hole()
  # run_spherical_cube()

  run_icosahedron_spherical()
  run_icosahedron()
