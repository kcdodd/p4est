import networkx as nx
import numpy as np
import sys
import pyvista as pv
from p4est import (
  P4est,
  QuadMesh )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def unit_square():
  verts = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0] ])

  cells = np.array([
    [[0, 1], [2, 3]] ])

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

  verts = np.zeros((13, 3), dtype = np.float64)

  i = np.arange(6)
  t = np.pi*i/3

  verts[1::2, 0] = r1*np.cos(t)
  verts[1::2, 1] = r1*np.sin(t)

  verts[2::2, 0] = r2*np.cos(t + np.pi / 6)
  verts[2::2, 1] = r2*np.sin(t + np.pi / 6)

  cells = np.array([
    [[0, 1], [3, 2]],
    [[0, 3], [5, 4]],
    [[5, 6], [0, 7]],
    [[8, 7], [9, 0]],
    [[9, 0], [10, 11]],
    [[12, 1], [11, 0]] ],
    dtype = np.int32)

  return QuadMesh(
    verts = verts,
    cells = cells )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def periodic_stack():
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

  vert_nodes = -np.ones(len(verts), dtype = np.int32)
  vert_nodes[[0,-2]] = 0
  vert_nodes[[1,-1]] = 1

  cells = np.array([
    [[0, 1], [2, 3]],
    [[2, 3], [4, 5]],
    [[4, 5], [6, 7]], ],
    dtype = np.int32)

  return QuadMesh(
    verts = verts,
    cells = cells,
    vert_nodes = vert_nodes )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def icosphere():

  c3 = np.cos(np.pi/3)
  s3 = np.sin(np.pi/3)

  # verts = np.array([
  #   [ 0.0 + c3,    s3,  0.0 ],
  #   [ 1.0 + c3,    s3,  0.0 ],
  #   [ 2.0 + c3,    s3,  0.0 ],
  #   [ 3.0 + c3,    s3,  0.0 ],
  #   [ 4.0 + c3,    s3,  0.0 ],
  #   [ 0.0,  0.0,  0.0 ],
  #   [ 1.0,  0.0,  0.0 ],
  #   [ 2.0,  0.0,  0.0 ],
  #   [ 3.0,  0.0,  0.0 ],
  #   [ 4.0,  0.0,  0.0 ],
  #   [ 5.0,  0.0,  0.0 ],
  #   [ 0.0 + c3, -  s3,  0.0 ],
  #   [ 1.0 + c3, -  s3,  0.0 ],
  #   [ 2.0 + c3, -  s3,  0.0 ],
  #   [ 3.0 + c3, -  s3,  0.0 ],
  #   [ 4.0 + c3, -  s3,  0.0 ],
  #   [ 5.0 + c3, -  s3,  0.0 ],
  #   [ 0.0 + 2*c3, -2*s3,  0.0 ],
  #   [ 1.0 + 2*c3, -2*s3,  0.0 ],
  #   [ 2.0 + 2*c3, -2*s3,  0.0 ],
  #   [ 3.0 + 2*c3, -2*s3,  0.0 ],
  #   [ 4.0 + 2*c3, -2*s3,  0.0 ] ])

  theta = (2*np.pi/10)*np.arange(12)

  phi = np.array([
    -0.5*np.pi,
    - np.arctan(0.5),
    np.arctan(0.5),
    0.5*np.pi ])

  verts = np.zeros((22,3))
  verts[:,2] = 1.0


  verts[:5,0] = theta[1:10:2]
  verts[:5,1] = phi[0]

  verts[5:11,0] = theta[:11:2]
  verts[5:11,1] = phi[1]

  verts[11:17,0] = theta[1:12:2]
  verts[11:17,1] = phi[2]

  verts[17:,0] = theta[2:11:2]
  verts[17:,1] = phi[3]

  verts = trans_sphere_to_cart(verts)

  # verts = np.array([
  #   [ 0.0 + c3,    s3,  0.0 ],
  #   [ 1.0 + c3,    s3,  0.0 ],
  #   [ 2.0 + c3,    s3,  0.0 ],
  #   [ 3.0 + c3,    s3,  0.0 ],
  #   [ 4.0 + c3,    s3,  0.0 ],
  #   [ 0.0,  0.0,  0.0 ],
  #   [ 1.0,  0.0,  0.0 ],
  #   [ 2.0,  0.0,  0.0 ],
  #   [ 3.0,  0.0,  0.0 ],
  #   [ 4.0,  0.0,  0.0 ],
  #   [ 5.0,  0.0,  0.0 ],
  #   [ 0.0 + c3, -  s3,  0.0 ],
  #   [ 1.0 + c3, -  s3,  0.0 ],
  #   [ 2.0 + c3, -  s3,  0.0 ],
  #   [ 3.0 + c3, -  s3,  0.0 ],
  #   [ 4.0 + c3, -  s3,  0.0 ],
  #   [ 5.0 + c3, -  s3,  0.0 ],
  #   [ 0.0 + 2*c3, -2*s3,  0.0 ],
  #   [ 1.0 + 2*c3, -2*s3,  0.0 ],
  #   [ 2.0 + 2*c3, -2*s3,  0.0 ],
  #   [ 3.0 + 2*c3, -2*s3,  0.0 ],
  #   [ 4.0 + 2*c3, -2*s3,  0.0 ] ])

  vert_nodes = -np.ones(len(verts), dtype = np.int32)
  vert_nodes[[0,1,2,3,4]] = 0
  vert_nodes[[5,10]] = 1
  vert_nodes[[11,16]] = 2
  vert_nodes[[17,18,19,20,21]] = 3

  cells = np.array([
    [[5,  11], [0,  6]],
    [[11, 17], [6, 12]],
    [[6,  12], [1,  7]],
    [[12, 18], [7, 13]],
    [[7,  13], [2,  8]],
    [[13, 19], [8, 14]],
    [[8,  14], [3,  9]],
    [[14, 20], [9, 15]],
    [[9,  15], [4, 10]],
    [[15, 21], [10, 16]]],
    dtype = np.int32 )

  return QuadMesh(
    verts = verts,
    cells = cells,
    vert_nodes = vert_nodes)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def trans_sphere_to_cart(coords):
  xyz = np.zeros_like(coords)
  xyz[:,0] = coords[:,2]*np.cos(coords[:,0])*np.cos(coords[:,1])
  xyz[:,1] = coords[:,2]*np.sin(coords[:,0])*np.cos(coords[:,1])
  xyz[:,2] = coords[:,2]*np.sin(coords[:,1])

  return xyz

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_quad_slerp(verts, uv):

  # adapted from p4est_geometry_icosahedron_X
  n0 = verts[:,0,0]
  n1 = verts[:,0,1]
  n2 = verts[:,1,0]
  n3 = verts[:,1,1]

  eta_x = uv[:,0]
  eta_y = uv[:,1]

  # apply slerp
  # - between n0 and n1
  # - between n2 and n3
  #
  norme2 = np.sum(n0*n0, axis = 1)
  theta1 = np.arccos(np.sum(n0 * n1, axis = 1) / norme2)

  c0 = np.sin((1.0 - eta_x) * theta1) / np.sin(theta1)
  c1 = np.sin(eta_x * theta1) / np.sin(theta1)

  xyz01 = c0[:,None] * n0 + c1[:,None] * n1
  xyz23 = c0[:,None] * n2 + c1[:,None] * n3

  # apply slerp
  # - between xyz01 and xyz23
  theta2 = np.arccos(np.sum(xyz01 * xyz23, axis = 1) / norme2 )

  a0 = np.sin((1.0 - eta_y) * theta2) / np.sin(theta2)
  a1 = np.sin(eta_y * theta2) / np.sin(theta2)

  return a0[:,None] * xyz01 + a1[:,None] * xyz23

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


  for i in range(len(mesh.node_cells_offset)-1):
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

  faces = np.empty((nc, 5), dtype = mesh.cells.dtype)
  faces[:,0] = 4
  faces[:nc,1] = idx
  faces[:nc,2] = idx + nc
  faces[:nc,3] = idx + 2*nc
  faces[:nc,4] = idx + 3*nc

  p.add_mesh(
    pv.PolyData(verts, faces = faces.ravel()),
    # scalars = np.arange(len(mesh.verts)),
    # scalars = np.arange(len(mesh.cells)),
    show_edges = True,
    line_width = 1,
    point_size = 3 )


  p.add_axes()
  p.add_cursor(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
  p.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# mesh = star()
# mesh = periodic_stack()
mesh = icosphere()


grid = P4est(
  mesh = mesh,
  min_level = 0)

print("leaf_info")
print(grid.leaf_info)

print("centers")
print(grid.leaf_coord(uv = (0.5, 0.5)))

plot_mesh(mesh)

for r in range(3):
  grid.leaf_info['refine'] = 1
  grid.refine()


plot_grid(grid, interp = interp_quad_slerp)

# print('rank', grid.comm.rank, len(grid.leaf_info))
# print('rank', grid.comm.rank, grid.leaf_info)



