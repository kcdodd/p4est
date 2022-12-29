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
def cube(length = 1.0):
  half = 0.5*length

  verts = np.stack(
    np.meshgrid(
      [-half, half],
      [-half, half],
      [-half, half],
      indexing = 'ij'),
    axis = -1).transpose().reshape(-1, 3)

  print(verts)

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
def icosahedron_spherical(radius = 1.0):

  c3 = np.cos(np.pi/3)
  s3 = np.sin(np.pi/3)

  theta = np.linspace(-np.pi, np.pi, 6)
  dtheta = theta[1] - theta[0]

  phi = np.array([
    -0.5*np.pi,
    - np.arctan(0.5),
    np.arctan(0.5),
    0.5*np.pi ])

  verts = np.zeros((22,3))

  verts[:5,0] = theta[:5]
  verts[:5,1] = phi[0]

  verts[5:11,0] = theta - 0.5*dtheta
  verts[5:11,1] = phi[1]

  verts[11:17,0] = theta
  verts[11:17,1] = phi[2]

  verts[17:,0] = theta[:5] + 0.5*dtheta
  verts[17:,1] = phi[3]

  verts[:,2] = radius

  # verts = trans_sphere_to_cart(verts)

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
def icosahedron(radius = 1.0):
  mesh = icosahedron_spherical(radius = radius)

  return QuadMesh(
    verts = trans_sphere_to_cart(mesh.verts),
    cells = mesh.cells,
    vert_nodes = mesh.vert_nodes)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def trans_sphere_to_cart(coords):
  """Transforms coordinates from spherical to cartesian

  Parameters
  ----------
  coords : array with shape = (NV, 3)

    The coordinates are assumed to be (in order):
    * azimuthal angle [-pi, pi]
    * polar angle [-pi/2, pi/2]
    * radius


  """

  theta = coords[...,0]
  phi = coords[...,1]
  r = coords[...,2]

  xyz = np.zeros_like(coords)
  xyz[...,0] = r*np.cos(theta)*np.cos(phi)
  xyz[...,1] = r*np.sin(theta)*np.cos(phi)
  xyz[...,2] = r*np.sin(phi)

  return xyz

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_slerp(eta, x0, x1):
  """Spherical linear interpolation
  """
  _x0 = np.linalg.norm(x0, axis = -1)
  _x1 = np.linalg.norm(x1, axis = -1)

  cos_theta = np.sum(x0*x1, axis = -1) / (_x0 * _x1)
  sin_theta = np.sqrt(1.0 - cos_theta**2)
  theta = np.arccos(cos_theta)

  c0 = np.sin((1.0 - eta) * theta) / sin_theta
  c1 = np.sin(eta * theta) / sin_theta

  return c0[...,None] * x0 + c1[...,None] * x1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_slerp_quad(verts, uv):
  """Spherical linear interpolation of quadrilateral vertices
  """
  return interp_slerp(
    eta = uv[:,1],
    x0 = interp_slerp(
      eta = uv[:,0],
      x0 = verts[:,0,0],
      x1 = verts[:,0,1]),
    x1 = interp_slerp(
      eta = uv[:,0],
      x0 = verts[:,1,0],
      x1 = verts[:,1,1]) )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_sphere_to_cart_slerp(verts, uv):
  return interp_slerp_quad(
    verts = trans_sphere_to_cart(verts),
    uv = uv )

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

  faces = np.empty((nc, 5), dtype = np.int32)
  faces[:,0] = 4
  faces[:nc,1] = idx
  faces[:nc,2] = idx + nc
  faces[:nc,3] = idx + 2*nc
  faces[:nc,4] = idx + 3*nc

  p.add_mesh(
    pv.PolyData(verts, faces = faces.ravel()),
    scalars = grid.leaf_info['root'],
    show_edges = True,
    line_width = 1,
    point_size = 3 )


  p.add_axes()
  p.add_cursor(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
  p.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# mesh = star()
# mesh = periodic_stack()
# mesh = icosahedron()
mesh = icosahedron_spherical()
# mesh = cube()


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


plot_grid(grid, interp = interp_sphere_to_cart_slerp)

# print('rank', grid.comm.rank, len(grid.leaf_info))
# print('rank', grid.comm.rank, grid.leaf_info)



