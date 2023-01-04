
import numpy as np
from scipy.interpolate import (
  RectBivariateSpline)
import sys
import time
from pathlib import Path
from PIL import (
  ImageOps,
  Image )
import pyvista as pv
from p4est import (
  P4est,
  QuadMesh )

from mpi4py.util.dtlib import from_numpy_dtype

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cube(length = 1.0):
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
def interpgrid_2D_to_3D():
  im1 = Image.open(Path(__file__).parent / 'earth_no_clouds.jpg')
  im2 = ImageOps.grayscale(im1)
  gray_image = np.array(im2)
  scaled_image = gray_image / 255

  u = np.linspace(0 , np.pi, scaled_image.shape[0])
  v = np.linspace(-np.pi, np.pi, scaled_image.shape[1])
  r = scaled_image

  f = RectBivariateSpline(u , v , r, kx = 1, ky = 1 )

  return f

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
def trans_cart_to_sphere(xyz):
  """Transforms coordinates from spherical to cartesian

  Parameters
  ----------
  coords : array with shape = (NV, 3)

    The coordinates are assumed to be (in order):
    * azimuthal angle [-pi, pi]
    * polar angle [-pi/2, pi/2]
    * radius


  """

  x = xyz[...,0]
  y = xyz[...,1]
  z = xyz[...,2]
  r = np.linalg.norm(xyz, axis = -1)
  tpr = np.zeros_like(xyz)
  tpr[...,0] = np.arctan2(y, x)
  tpr[...,1] = np.arcsin(z / r)
  tpr[...,2] = r

  return tpr

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def interp_slerp(eta, x0, x1):
  """Spherical linear interpolation
  """
  _x0 = np.linalg.norm(x0, axis = -1)
  _x1 = np.linalg.norm(x1, axis = -1)

  cos_theta = np.sum(x0*x1, axis = -1) / (_x0 * _x1)
 # sin_theta = np.sqrt(1.0 - cos_theta**2)
  theta = np.arccos(cos_theta)
  sin_theta = np.sin(theta)
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
def mpi_ghost_exchange(grid, local_value):

  # exchange process neighbor information
  sendbuf = np.ascontiguousarray(local_value[grid.rank_mirrors.flat.idx])
  recvbuf = np.empty((len(grid.rank_ghosts.flat),), dtype = local_value.dtype)

  mpi_datatype = from_numpy_dtype(local_value.dtype)

  grid.comm.Alltoallv(
    sendbuf = [
      sendbuf, (
        # counts
        grid.rank_mirrors.row_counts,
        # displs
        grid.rank_mirrors.row_idx[:-1] ),
      mpi_datatype ],
    recvbuf = [
      recvbuf, (
        grid.rank_ghosts.row_counts,
        grid.rank_ghosts.row_idx[:-1] ),
      mpi_datatype ])

  return np.concatenate([local_value, recvbuf])

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_grid(grid, interp = None, scalars = None):
  scale = 0.99
  _scale = 1.0 - scale


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
  pv.set_plot_theme('paraview')
  p = pv.Plotter()
  p.add_mesh(
     pv.PolyData(verts, faces = faces.ravel()),
     scalars = grid.leaf_info.root if scalars is None else scalars,
     show_edges = False,
     line_width = 1,
     point_size = 3 )

  p.show()

  return verts,faces

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mesh = cube()
f = interpgrid_2D_to_3D()
tol = 0.05


grid = P4est(
  mesh = mesh,
  min_level = 0)

for p in range(4):
  grid.leaf_info.adapt = 1
  grid.adapt()


for r in range(6):

  points = trans_cart_to_sphere(grid.leaf_coord(uv = (0.5,0.5), interp = interp_slerp_quad))
  points[...,1] = np.pi / 2 - points[...,1]
  local_value = f(*points.transpose(1,0)[:2][::-1], grid = False)

  value = mpi_ghost_exchange(grid, local_value)

  value_adj = value[grid.leaf_info.cell_adj]
  d_value = np.abs(local_value[:,None,None,None] - value_adj).max(axis = (1,2,3))

  refine = d_value > tol
  grid.leaf_info.adapt = refine

  if not np.any(refine):
    break

  grid.adapt()


points = trans_cart_to_sphere(grid.leaf_coord(uv = (0.5,0.5), interp = interp_slerp_quad))
points[...,1] = np.pi / 2 - points[...,1]
value = f(*points.transpose(1,0)[:2][::-1], grid = False)

plot_grid(grid, interp = interp_slerp_quad, scalars = value)




