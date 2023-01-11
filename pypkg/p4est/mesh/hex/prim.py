import numpy as np
from ...geom import (
  trans_sphere_to_cart )
from .mesh import (
  HexMesh,
  HexMeshSpherical,
  HexMeshCartesianSpherical )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cube(length = 1.0):
  """Factory method to create the volume of a cube

  Parameters
  ----------
  length : float

  Returns
  -------
  HexMesh
  """

  half = 0.5*length

  verts = np.stack(
    np.meshgrid(
      [-half, half],
      [-half, half],
      [-half, half],
      indexing = 'ij'),
    axis = -1).transpose(2,1,0,3).reshape(-1, 3)

  cells = np.arange(8).reshape(1,2,2,2)

  return HexMesh(
    verts = verts,
    cells = cells )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def spherical_cube(length = 1.0):
  """Factory method to create the volume of a cube

  Parameters
  ----------
  length : float

  Returns
  -------
  HexMesh
  """

  half = 0.5*length

  verts = np.stack(
    np.meshgrid(
      [-half, half],
      [-half, half],
      [-half, half],
      indexing = 'ij'),
    axis = -1).transpose(2,1,0,3).reshape(-1, 3)

  cells = np.arange(8).reshape(1,2,2,2)

  return HexMeshCartesianSpherical(
    verts = verts,
    cells = cells )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def icosahedron_spherical_shell(r1 = 0.5, r2 = 1.0):
  """Factory method to create an icosahedron shell in spherical coordinates

  Parameters
  ----------
  radius : float
    Outer radius of points

  Returns
  -------
  HexMesh
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

  _verts = np.zeros((22,3))

  _verts[:5,0] = phi[0]
  _verts[:5,1] = theta[:5]

  _verts[5:11,0] = phi[1]
  _verts[5:11,1] = theta - 0.5*dtheta

  _verts[11:17,0] = phi[2]
  _verts[11:17,1] = theta

  _verts[17:,0] = phi[3]
  _verts[17:,1] = theta[:5] + 0.5*dtheta


  verts = np.concatenate([
    _verts + np.array([0.0, 0.0, r1])[None,:],
    _verts + np.array([0.0, 0.0, r2])[None,:] ])

  _vert_nodes = -np.ones(len(_verts), dtype = np.int32)
  _vert_nodes[[0,1,2,3,4]] = 0
  _vert_nodes[[5,10]] = 2
  _vert_nodes[[11,16]] = 3
  _vert_nodes[[17,18,19,20,21]] = 1

  vert_nodes = np.concatenate([
    _vert_nodes,
    np.where(_vert_nodes == -1, _vert_nodes, _vert_nodes + 4) ])

  _cells = np.array([
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

  cells = np.stack([
    _cells,
    _cells + len(_verts)],
    axis = 1 )

  return HexMeshSpherical(
    verts = verts,
    cells = cells,
    vert_nodes = vert_nodes)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def icosahedron_shell(r1 = 0.5, r2 = 1.0):
  """Factory method to create an icosahedron in cartesian coordinates

  Parameters
  ----------
  radius : float
    Outer radius of points

  Returns
  -------
  HexMesh
  """
  mesh = icosahedron_spherical_shell(r1 = r1, r2 = r2)

  return HexMeshCartesianSpherical(
    verts = trans_sphere_to_cart(mesh.verts),
    cells = mesh.cells,
    vert_nodes = mesh.vert_nodes)