import numpy as np
from ...geom import trans_sphere_to_cart
from .geom import (
  QuadLinear,
  QuadSpherical,
  QuadCartesianSpherical )
from .base import QuadMesh

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def unit_square() -> QuadMesh:
  """Factory method to create a unit square
  """

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
def cube(length : float = 1.0) -> QuadMesh:
  """Factory method to create the surface of a cube

  Parameters
  ----------
  length :
    Length of sides
  """

  half = 0.5*length

  verts = np.stack(
    np.meshgrid(
      [-half, half],
      [-half, half],
      [-half, half],
      indexing = 'ij'),
    axis = -1).transpose(2,1,0,3).reshape(-1, 3)

  #Cell with vertex ordering [V0, V1] , [V2, V3]
  cells = np.array([
    [[0, 1], [4, 5]], #Origin Cell

    [[1, 3], [5, 7]],  #Right of Origin Cell

    [[2, 0], [6, 4]],  #Left of Origin Cell

    [[3, 2], [7, 6]],  #Opposite of Origin Cell

    [[1, 3], [0, 2]],  #Bottom Cell

    [[5, 7], [4, 6]]])  #Top Cell

  return QuadMesh(
    verts = verts,
    cells = cells )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def spherical_cube(length : float = 1.0) -> QuadMesh:
  """Factory method to create the surface of a cube

  Parameters
  ----------
  length : float
    Length of sides

  """

  half = 0.5*length

  verts = np.stack(
    np.meshgrid(
      [-half, half],
      [-half, half],
      [-half, half],
      indexing = 'ij'),
    axis = -1).transpose(2,1,0,3).reshape(-1, 3)

  #Cell with vertex ordering [V0, V1] , [V2, V3]
  cells = np.array([
    [[0, 1], [4, 5]], #Origin Cell

    [[1, 3], [5, 7]],  #Right of Origin Cell

    [[2, 0], [6, 4]],  #Left of Origin Cell

    [[3, 2], [7, 6]],  #Opposite of Origin Cell

    [[1, 3], [0, 2]],  #Bottom Cell

    [[5, 7], [4, 6]]])  #Top Cell

  return QuadMesh(
    verts = verts,
    cells = cells,
    geoms = QuadCartesianSpherical() )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def stack_o_squares() -> QuadMesh:
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
def star(r1 : float = 1.0, r2 : float = 1.5) -> QuadMesh:
  """Factory method to create a 6 point star

  Parameters
  ----------
  r1 :
    Outer radius of points.
  r2 :
    Inner radius where points merge.

  """

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
def periodic_stack() -> QuadMesh:
  """Factory method to create a periodic stack of unit squares
  """

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
def icosahedron_golden() -> QuadMesh:
  """Factory method to create an icosahedron using golden ratio
  """

  c3 = np.cos(np.pi/3)
  s3 = np.sin(np.pi/3)

  verts = np.array([
    [ 0.0 + c3,    s3,  0.0 ],
    [ 1.0 + c3,    s3,  0.0 ],
    [ 2.0 + c3,    s3,  0.0 ],
    [ 3.0 + c3,    s3,  0.0 ],
    [ 4.0 + c3,    s3,  0.0 ],
    [ 0.0,  0.0,  0.0 ],
    [ 1.0,  0.0,  0.0 ],
    [ 2.0,  0.0,  0.0 ],
    [ 3.0,  0.0,  0.0 ],
    [ 4.0,  0.0,  0.0 ],
    [ 5.0,  0.0,  0.0 ],
    [ 0.0 + c3, -  s3,  0.0 ],
    [ 1.0 + c3, -  s3,  0.0 ],
    [ 2.0 + c3, -  s3,  0.0 ],
    [ 3.0 + c3, -  s3,  0.0 ],
    [ 4.0 + c3, -  s3,  0.0 ],
    [ 5.0 + c3, -  s3,  0.0 ],
    [ 0.0 + 2*c3, -2*s3,  0.0 ],
    [ 1.0 + 2*c3, -2*s3,  0.0 ],
    [ 2.0 + 2*c3, -2*s3,  0.0 ],
    [ 3.0 + 2*c3, -2*s3,  0.0 ],
    [ 4.0 + 2*c3, -2*s3,  0.0 ] ])

  vert_nodes = -np.ones(len(verts), dtype = np.int32)
  vert_nodes[[0,1,2,3,4]] = 0
  vert_nodes[[5,10]] = 2
  vert_nodes[[11,16]] = 3
  vert_nodes[[17,18,19,20,21]] = 1

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
def icosahedron_spherical(radius : float = 1.0) -> QuadMesh:
  """Factory method to create an icosahedron in spherical coordinates

  Parameters
  ----------
  radius :
    Outer radius of points
  """

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

  verts[:5,0] = phi[0]
  verts[:5,1] = theta[:5]

  verts[5:11,0] = phi[1]
  verts[5:11,1] = theta - 0.5*dtheta

  verts[11:17,0] = phi[2]
  verts[11:17,1] = theta

  verts[17:,0] = phi[3]
  verts[17:,1] = theta[:5] + 0.5*dtheta

  verts[:,2] = radius

  vert_nodes = -np.ones(len(verts), dtype = np.int32)
  vert_nodes[[0,1,2,3,4]] = 0
  vert_nodes[[5,10]] = 2
  vert_nodes[[11,16]] = 3
  vert_nodes[[17,18,19,20,21]] = 1

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
    vert_nodes = vert_nodes,
    geoms = QuadSpherical() )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def icosahedron(radius : float = 1.0) -> QuadMesh:
  """Factory method to create an icosahedron in cartesian coordinates

  Parameters
  ----------
  radius : float
    Outer radius of points
  """
  mesh = icosahedron_spherical(radius = radius)

  return QuadMesh(
    verts = trans_sphere_to_cart(mesh.verts),
    cells = mesh.cells,
    vert_nodes = mesh.vert_nodes,
    geoms = QuadCartesianSpherical() )
