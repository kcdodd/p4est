import networkx as nx
import numpy as np
import sys
import pyvista as pv
from p4est import (
  P4est,
  QuadMesh )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def unit_square():
  verts = verts = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0] ])

  cells = verts = np.array([
    [0, 1, 2, 3] ])

  return QuadMesh(
    verts = verts,
    cells = cells )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def stack_o_squares():
  verts = np.array([
    [0, 0, 0],
    [1, 0, 0],

    [0, 1, 0],
    [1, 1, 0],

    [0, 2, 0],
    [1, 2, 0],

    [0, 3, 0],
    [1, 3, 0], ],
    dtype = np.float64)

  cells = np.array([
    [[0, 1], [2, 3]],
    [[2, 3], [4, 5]],
    [[4, 5], [6, 7]], ],
    dtype = np.int32)

  return QuadMesh(
    verts = verts,
    cells = cells )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def star(r1 = 1.0, r2 = 1.5):
  cells = np.array([
    [[0, 1], [3, 2]],
    [[0, 3], [5, 4]],
    [[5, 6], [0, 7]],
    [[8, 7], [9, 0]],
    [[9, 0], [10, 11]],
    [[12, 1], [11, 0]] ],
    dtype = np.int32)

  verts = np.zeros((13, 3), dtype = np.float64)

  i = np.arange(6)
  t = np.pi*i/3
  pi = 4.0 * np.arctan (1.0)

  verts[1::2, 0] = r1*np.cos(i * pi / 3)
  verts[1::2, 1] = r1*np.sin(i * pi / 3)

  verts[2::2, 0] = r2*np.cos((i + .5) * pi / 3)
  verts[2::2, 1] = r2*np.sin((i + .5) * pi / 3)

  return QuadMesh(
    verts = verts,
    cells = cells )

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

  p.add_mesh(
    pv.PolyData(mesh.verts, faces = faces.ravel()),
    # scalars = np.arange(len(mesh.verts)),
    scalars = np.arange(len(mesh.cells)),
    show_edges = True,
    line_width = 1,
    point_size = 3 )

  p.add_points(
    mesh.verts,
    point_size = 7,
    color = 'red',
    opacity = 0.75 )

  p.add_point_labels(
    mesh.verts,
    labels = [str(i) for i in range(len(mesh.verts))],
    text_color = 'yellow',
    font_size = 30,
    fill_shape = False )

  p.add_axes()
  p.add_cursor(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
  p.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

mesh = star()

plot_mesh(mesh)

grid = P4est(
  mesh = mesh,
  min_level = 0)

print("leaf_info")
print(grid.leaf_info)

print("centers")
print(grid.leaf_coord(uv = (0.5, 0.5)))

# grid.leaf_info['refine'] = 1
# grid.refine()
# print(grid.leaf_coord(uv = (0.5, 0.5)))

# print('rank', grid.comm.rank, len(grid.leaf_info))
# print('rank', grid.comm.rank, grid.leaf_info)



