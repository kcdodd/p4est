import numpy as np
from ...geom import trans_sphere_to_cart
from .geom import (
  HexLinear,
  HexSpherical,
  HexCartesianSpherical)
from .base import HexMesh


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cube(length : float = 1.0) -> HexMesh:
  """Factory method to create the volume of a cube

  .. figure:: ../img/Mesh_Cube.png
    :width: 90%
    :align: center

    Mesh Volume Cube.

  .. figure:: ../img/3D_Cube.png
    :width: 90%
    :align: center

    Volume Cube.

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

  cells = np.arange(8).reshape(1,2,2,2)

  return HexMesh(
    verts = verts,
    cells = cells )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def spherical_cube_shell(r1 : float = 0.5, r2 : float = 1.0) -> HexMesh:
  """Factory method to create the volume of a spherical shell

  .. figure:: ../img/mesh_spherical_cube_shell.png
    :width: 90%
    :align: center

    Mesh Speherical Cube Shell.

  .. figure:: ../img/3D_Spherical_cube_Shell.png
    :width: 90%
    :align: center

    Spherical Cube Shell.

  Parameters
  ----------
  r1 :
    Inner radius
  r2 :
    Outer radius
  """

  # half-length of cube edges for distance (radius) to the vertices
  # r = (l**2 + l**2 + l**2)**0.5 = l * 3**0.5
  l1 = r1 / 3**0.5
  l2 = r2 / 3**0.5

  inner = np.stack(
    np.meshgrid(
      [-l1, l1],
      [-l1, l1],
      [-l1, l1],
      indexing = 'ij'),
    axis = -1).transpose(2,1,0,3).reshape(-1, 3)

  outer = np.stack(
    np.meshgrid(
      [-l2, l2],
      [-l2, l2],
      [-l2, l2],
      indexing = 'ij'),
    axis = -1).transpose(2,1,0,3).reshape(-1, 3)

  verts = np.concatenate([inner, outer])

  cells = np.array([
      # -x
      [0, 2, 4, 6, 8, 10, 12, 14],
      # +x
      [1, 3, 5, 7, 9, 11, 13, 15],
      # -y
      [0, 1, 4, 5, 8, 9, 12, 13],
      # +y
      [2, 3, 6, 7, 10, 11, 14, 15],
      # -z
      [0, 1, 2, 3, 8, 9, 10, 11],
      # +z
      [4, 5, 6, 7, 12, 13, 14, 15] ],
    dtype = np.int32 ).reshape(-1, 2,2,2)

  return HexMesh(
    verts = verts,
    cells = cells,
    geoms = HexCartesianSpherical())

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def spherical_cube(r1 : float = 0.5, r2 : float = 1.0) -> HexMesh:
  """Factory method to create the volume of a sphere

  .. figure:: ../img/mesh_spherical_cube_shell.png
    :width: 90%
    :align: center

    Mesh Speherical Cube Shell.

  .. figure:: ../img/Spherical_Cube.png
    :width: 90%
    :align: center

    Spherical Cube Refined.

  Parameters
  ----------
  r1 :
    Inner radius
  r2 :
    Outer radius
  """

  # half-length of cube edges for distance (radius) to the vertices
  # r = (l**2 + l**2 + l**2)**0.5 = l * 3**0.5
  l1 = r1 / 3**0.5
  l2 = r2 / 3**0.5

  inner = np.stack(
    np.meshgrid(
      [-l1, l1],
      [-l1, l1],
      [-l1, l1],
      indexing = 'ij'),
    axis = -1).transpose(2,1,0,3).reshape(-1, 3)

  outer = np.stack(
    np.meshgrid(
      [-l2, l2],
      [-l2, l2],
      [-l2, l2],
      indexing = 'ij'),
    axis = -1).transpose(2,1,0,3).reshape(-1, 3)

  verts = np.concatenate([inner, outer])

  cells = np.array([
      [0, 1, 2, 3, 4, 5, 6, 7],
      # -x
      [0, 2, 4, 6, 8, 10, 12, 14],
      # +x
      [1, 3, 5, 7, 9, 11, 13, 15],
      # -y
      [0, 1, 4, 5, 8, 9, 12, 13],
      # +y
      [2, 3, 6, 7, 10, 11, 14, 15],
      # -z
      [0, 1, 2, 3, 8, 9, 10, 11],
      # +z
      [4, 5, 6, 7, 12, 13, 14, 15] ],
    dtype = np.int32 ).reshape(-1, 2,2,2)

  return HexMesh(
    verts = verts,
    cells = cells,
    geoms = [HexLinear(), HexCartesianSpherical()],
    vert_geom = [0]*8 + [1]*8)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def slab_spherical_cube_hole(
  lx : float = 0.5,
  ly : float = 0.5,
  lz : float = 1.0,
  z : float = 0.25,
  r : float = 0.25 ) \
  -> HexMesh:
  """Factory method to create the volume of slab with spherical hole

  Parameters
  ----------
  lx, ly, lz :
    Length of slab sides
  z :
    z-offset of hole
  r :
    Radius of hole

  """

  # half-length of cube edges for distance (radius) to the vertices
  # r = (l**2 + l**2 + l**2)**0.5 = l * 3**0.5
  l1 = r / 3**0.5

  inner = np.stack(
    np.meshgrid(
      [-l1, l1],
      [-l1, l1],
      [-l1, l1],
      indexing = 'ij'),
    axis = -1).transpose(2,1,0,3).reshape(-1, 3)

  inner[:,2] += z

  outer = np.stack(
    np.meshgrid(
      [-lx, lx],
      [-ly, ly],
      [-lz, lz],
      indexing = 'ij'),
    axis = -1).transpose(2,1,0,3).reshape(-1, 3)

  verts = np.concatenate([inner, outer])

  cells = np.array([
      # -x
      [0, 2, 4, 6, 8, 10, 12, 14],
      # +x
      [1, 3, 5, 7, 9, 11, 13, 15],
      # -y
      [0, 1, 4, 5, 8, 9, 12, 13],
      # +y
      [2, 3, 6, 7, 10, 11, 14, 15],
      # -z
      [0, 1, 2, 3, 8, 9, 10, 11],
      # +z
      [4, 5, 6, 7, 12, 13, 14, 15] ],
    dtype = np.int32 ).reshape(-1, 2,2,2)

  return HexMesh(
    verts = verts,
    cells = cells,
    geoms = [HexCartesianSpherical(origin = (0,0,z)), HexLinear()],
    vert_geom = [0]*8 + [1]*8)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def icosahedron_spherical_shell(r1 : float = 0.5, r2 : float = 1.0) -> HexMesh:
  """Factory method to create an icosahedron shell in spherical coordinates

  Parameters
  ----------
  r1 :
    Inner radius
  r2 :
    Outer radius
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

  return HexMesh(
    verts = verts,
    cells = cells,
    vert_nodes = vert_nodes,
    geoms = HexSpherical() )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def icosahedron_shell(r1 : float = 0.5, r2 : float = 1.0) -> HexMesh:
  """Factory method to create an icosahedron in cartesian coordinates

  .. figure:: ../img/mesh_Icosahedron_Spherical.png
    :width: 90%
    :align: center

    Mesh Speherical Shell.

  .. figure:: ../img/3D_Icosahedron_Spherical.png
    :width: 90%
    :align: center

    Icosahedron Shell.

  Parameters
  ----------
  r1 :
    Inner radius
  r2 :
    Outer radius
  """
  mesh = icosahedron_spherical_shell(r1 = r1, r2 = r2)

  return HexMesh(
    verts = trans_sphere_to_cart(mesh.verts),
    cells = mesh.cells,
    vert_nodes = mesh.vert_nodes,
    geoms = HexCartesianSpherical() )